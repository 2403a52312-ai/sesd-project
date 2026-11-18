const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");
if (menuToggle && navLinks) {
  menuToggle.addEventListener("click", () => navLinks.classList.toggle("open"));
}

const jobText = document.getElementById("job_text");
const charCount = document.getElementById("charCount");
if (jobText && charCount) {
  const updateCount = () => charCount.textContent = `${jobText.value.length} characters`;
  jobText.addEventListener("input", updateCount);
  updateCount();
}
