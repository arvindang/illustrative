/**
 * PDF.js viewer for the sample output gallery.
 * Renders all pages of the sample PDF vertically.
 */
(function () {
  const PDF_URL = 'samples/20K_Leagues_Under_the_Sea_test_page.pdf';
  const SCALE = 2.0; // Render at 2x for crisp display

  const loadingEl = document.getElementById('pdf-loading');
  const pagesEl = document.getElementById('pdf-pages');

  // Load PDF.js from CDN
  const script = document.createElement('script');
  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs';
  script.type = 'module';

  // Use a module script to import and render
  const moduleScript = document.createElement('script');
  moduleScript.type = 'module';
  moduleScript.textContent = `
    import * as pdfjsLib from 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs';

    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs';

    const PDF_URL = '${PDF_URL}';
    const SCALE = ${SCALE};
    const loadingEl = document.getElementById('pdf-loading');
    const pagesEl = document.getElementById('pdf-pages');

    async function renderPDF() {
      try {
        const pdf = await pdfjsLib.getDocument(PDF_URL).promise;
        const numPages = pdf.numPages;

        loadingEl.classList.add('hidden');
        pagesEl.classList.remove('hidden');

        for (let i = 1; i <= numPages; i++) {
          const page = await pdf.getPage(i);
          const viewport = page.getViewport({ scale: SCALE });

          const canvas = document.createElement('canvas');
          canvas.width = viewport.width;
          canvas.height = viewport.height;

          const ctx = canvas.getContext('2d');
          await page.render({ canvasContext: ctx, viewport: viewport }).promise;

          // Page label
          const label = document.createElement('p');
          label.className = 'text-xs text-slate-600 mb-2';
          label.textContent = 'Page ' + i + ' of ' + numPages;

          const wrapper = document.createElement('div');
          wrapper.className = 'flex flex-col items-center';
          wrapper.appendChild(label);
          wrapper.appendChild(canvas);

          pagesEl.appendChild(wrapper);
        }
      } catch (err) {
        console.error('PDF render error:', err);
        loadingEl.innerHTML = '<p class="text-slate-500">Could not load PDF preview. <a href="${PDF_URL}" class="text-indigo-400 underline">Download it directly</a>.</p>';
      }
    }

    renderPDF();
  `;

  document.head.appendChild(moduleScript);
})();
