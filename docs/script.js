document.addEventListener("DOMContentLoaded", function() {
  const hero = document.getElementById("hero");
  const target = document.getElementById("start");

  if (hero && target) {
    hero.addEventListener("click", function(event) {
      event.preventDefault(); // blokuje domyślną akcję <a>
      const offset = 65;
      const topPosition = target.getBoundingClientRect().top + window.scrollY - offset;

      window.scrollTo({
        top: topPosition,
        behavior: "smooth"
      });
    });
  }
});
