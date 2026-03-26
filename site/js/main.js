/**
 * Main interactions for the Illustrative AI showcase site.
 * - Sticky nav with active section highlighting
 * - Mobile menu toggle
 * - Scroll-triggered fade-in animations
 * - Copy-to-clipboard for code blocks
 */
(function () {
  // ================================================================
  // Mobile menu toggle
  // ================================================================
  const menuBtn = document.getElementById('mobile-menu-btn');
  const mobileMenu = document.getElementById('mobile-menu');

  if (menuBtn && mobileMenu) {
    menuBtn.addEventListener('click', () => {
      mobileMenu.classList.toggle('hidden');
    });

    // Close menu when a link is clicked
    mobileMenu.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        mobileMenu.classList.add('hidden');
      });
    });
  }

  // ================================================================
  // Active nav link highlighting via Intersection Observer
  // ================================================================
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('.nav-link');

  const navObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          navLinks.forEach((link) => {
            if (link.getAttribute('href') === '#' + id) {
              link.classList.add('active');
            } else {
              link.classList.remove('active');
            }
          });
        }
      });
    },
    {
      rootMargin: '-20% 0px -70% 0px',
    }
  );

  sections.forEach((section) => navObserver.observe(section));

  // ================================================================
  // Scroll-triggered fade-in animations
  // ================================================================
  // Add .fade-in to key elements
  document
    .querySelectorAll(
      'section > div > h2, section > div > .grid, section > div > .bg-slate-900, section > div > .flex'
    )
    .forEach((el) => {
      el.classList.add('fade-in');
    });

  const fadeObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    },
    {
      threshold: 0.1,
    }
  );

  document.querySelectorAll('.fade-in').forEach((el) => fadeObserver.observe(el));
})();

// ================================================================
// Copy to clipboard
// ================================================================
function copyCode(button) {
  const code = button.getAttribute('data-code');
  if (!code) return;

  navigator.clipboard
    .writeText(code)
    .then(() => {
      const original = button.textContent;
      button.textContent = 'Copied!';
      setTimeout(() => {
        button.textContent = original;
      }, 2000);
    })
    .catch(() => {
      // Fallback
      button.textContent = 'Failed';
      setTimeout(() => {
        button.textContent = 'Copy';
      }, 2000);
    });
}
