// CardioAI front-end interactions

function showDashboardPage(index) {
  const pages = document.querySelectorAll('.dashboard-page');
  const tabs = document.querySelectorAll('.dashboard-tab');
  pages.forEach((p, i) => p.classList.toggle('active', i === index));
  tabs.forEach((t, i) => t.classList.toggle('active', i === index));
}

// Smooth scroll for in-page anchors (progressive enhancement)
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
});
