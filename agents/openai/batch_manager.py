"""
OpenAI Batch API lifecycle manager.

Handles the full batch workflow:
  1. Build JSONL from a list of BatchRequest objects
  2. Upload JSONL to OpenAI Files API
  3. Create a batch job
  4. Poll until complete (or timeout)
  5. Download and parse results

Supports both /v1/images/generations and /v1/images/edits endpoints.
"""
import asyncio
import io
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agents.openai.client import get_openai_client


@dataclass
class BatchRequest:
    """A single request within a batch job."""
    custom_id: str
    method: str = "POST"
    url: str = "/v1/images/generations"
    body: dict = field(default_factory=dict)


@dataclass
class BatchResult:
    """Parsed result for a single request in a completed batch."""
    custom_id: str
    status_code: int = 200
    b64_image: Optional[str] = None
    revised_prompt: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status_code == 200 and self.b64_image is not None


class OpenAIBatchManager:
    """
    Manages the OpenAI Batch API lifecycle for image generation jobs.

    Usage:
        manager = OpenAIBatchManager()
        results = await manager.run_batch(requests)
        for custom_id, result in results.items():
            if result.success:
                image_bytes = base64.b64decode(result.b64_image)
    """

    def __init__(self, poll_interval: int = 30, timeout: int = 86400):
        """
        Args:
            poll_interval: Seconds between batch status polls (default 30s).
            timeout: Max seconds to wait for batch completion (default 24h).
        """
        self.poll_interval = poll_interval
        self.timeout = timeout

    def _build_jsonl(self, requests: List[BatchRequest]) -> bytes:
        """Serialize a list of BatchRequest objects into JSONL bytes."""
        lines = []
        for req in requests:
            line = {
                "custom_id": req.custom_id,
                "method": req.method,
                "url": req.url,
                "body": req.body,
            }
            lines.append(json.dumps(line, separators=(",", ":")))
        return "\n".join(lines).encode("utf-8")

    async def submit_batch(self, requests: List[BatchRequest]) -> str:
        """
        Build JSONL, upload it, and create a batch job.

        Args:
            requests: List of BatchRequest objects to include in the batch.

        Returns:
            The batch job ID (e.g., "batch_abc123").
        """
        client = get_openai_client()
        jsonl_bytes = self._build_jsonl(requests)

        # Upload JSONL as an input file
        upload_file = await client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes)),
            purpose="batch",
        )
        print(f"   Uploaded batch input file: {upload_file.id} ({len(requests)} requests)")

        # Create the batch
        batch = await client.batches.create(
            input_file_id=upload_file.id,
            endpoint="/v1/images/generations",
            completion_window="24h",
        )
        print(f"   Created batch: {batch.id} (status: {batch.status})")
        return batch.id

    async def poll_until_complete(self, batch_id: str) -> str:
        """
        Poll the batch until it reaches a terminal state.

        Args:
            batch_id: The batch job ID to poll.

        Returns:
            The output_file_id for downloading results.

        Raises:
            TimeoutError: If the batch doesn't complete within self.timeout.
            RuntimeError: If the batch fails, is cancelled, or expires.
        """
        client = get_openai_client()
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {self.timeout}s"
                )

            batch = await client.batches.retrieve(batch_id)
            status = batch.status

            if status == "completed":
                completed = batch.request_counts.completed if batch.request_counts else "?"
                failed = batch.request_counts.failed if batch.request_counts else "?"
                print(f"   Batch {batch_id} completed: {completed} succeeded, {failed} failed")
                if not batch.output_file_id:
                    raise RuntimeError(f"Batch {batch_id} completed but has no output_file_id")
                return batch.output_file_id

            if status in ("failed", "cancelled", "expired"):
                errors = ""
                if batch.errors and batch.errors.data:
                    errors = "; ".join(
                        f"{e.code}: {e.message}" for e in batch.errors.data[:3]
                    )
                raise RuntimeError(
                    f"Batch {batch_id} ended with status '{status}'. Errors: {errors or 'none'}"
                )

            # Still in progress
            total = batch.request_counts.total if batch.request_counts else "?"
            completed = batch.request_counts.completed if batch.request_counts else "?"
            print(
                f"   Batch {batch_id}: {status} ({completed}/{total} done, "
                f"{int(elapsed)}s elapsed)"
            )
            await asyncio.sleep(self.poll_interval)

    async def download_results(self, output_file_id: str) -> List[BatchResult]:
        """
        Download and parse the output JSONL from a completed batch.

        Args:
            output_file_id: The file ID of the batch output.

        Returns:
            List of BatchResult objects.
        """
        client = get_openai_client()
        content = await client.files.content(output_file_id)
        raw_text = content.text

        results = []
        for line in raw_text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            response = entry.get("response", {})
            status_code = response.get("status_code", 500)
            body = response.get("body", {})

            result = BatchResult(custom_id=custom_id, status_code=status_code)

            if status_code == 200:
                # Extract image data from response
                data_list = body.get("data", [])
                if data_list:
                    result.b64_image = data_list[0].get("b64_json")
                    result.revised_prompt = data_list[0].get("revised_prompt")
            else:
                error_obj = body.get("error", {})
                result.error = error_obj.get("message", str(body))

            results.append(result)

        return results

    async def run_batch(self, requests: List[BatchRequest]) -> Dict[str, BatchResult]:
        """
        Full batch lifecycle: submit -> poll -> download -> return results.

        Args:
            requests: List of BatchRequest objects.

        Returns:
            Dict mapping custom_id to BatchResult.
        """
        if not requests:
            return {}

        print(f"\n--- OpenAI Batch: {len(requests)} requests ---")
        batch_id = await self.submit_batch(requests)
        output_file_id = await self.poll_until_complete(batch_id)
        results = await self.download_results(output_file_id)

        # Index by custom_id
        results_map = {}
        for result in results:
            results_map[result.custom_id] = result

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        print(f"   Batch complete: {succeeded} succeeded, {failed} failed")

        return results_map
