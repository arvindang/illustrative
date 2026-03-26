/**
 * Image carousel for the sample output gallery.
 * Navigates 10 pre-rendered WebP page images.
 */
(function () {
  const TOTAL_PAGES = 10;
  let current = 1;

  const img = document.getElementById('carousel-img');
  const counter = document.getElementById('carousel-counter');
  const prevBtn = document.getElementById('carousel-prev');
  const nextBtn = document.getElementById('carousel-next');

  if (!img || !counter) return;

  function padPage(n) {
    return String(n).padStart(2, '0');
  }

  function goTo(page) {
    if (page < 1 || page > TOTAL_PAGES) return;
    current = page;
    img.src = 'samples/pages/page-' + padPage(current) + '.webp';
    img.alt = 'Sample page ' + current + ' of ' + TOTAL_PAGES;
    counter.textContent = 'Page ' + current + ' of ' + TOTAL_PAGES;
    prevBtn.disabled = current === 1;
    nextBtn.disabled = current === TOTAL_PAGES;
    prevBtn.style.opacity = current === 1 ? '0.3' : '1';
    nextBtn.style.opacity = current === TOTAL_PAGES ? '0.3' : '1';
  }

  prevBtn.addEventListener('click', function () { goTo(current - 1); });
  nextBtn.addEventListener('click', function () { goTo(current + 1); });

  // Keyboard navigation
  document.addEventListener('keydown', function (e) {
    // Only handle arrows when carousel is in viewport
    var rect = img.getBoundingClientRect();
    if (rect.bottom < 0 || rect.top > window.innerHeight) return;

    if (e.key === 'ArrowLeft') { goTo(current - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { goTo(current + 1); e.preventDefault(); }
  });

  // Preload adjacent pages
  function preload(page) {
    if (page >= 1 && page <= TOTAL_PAGES) {
      var link = new Image();
      link.src = 'samples/pages/page-' + padPage(page) + '.webp';
    }
  }

  // Preload pages 2 and 3 on load
  preload(2);
  preload(3);

  // Initialize
  goTo(1);
})();
