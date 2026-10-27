/**
 * FIRMNet AI — Client-side logic
 * Navigation, form validation, file upload, loading states.
 */

document.addEventListener("DOMContentLoaded", () => {
  initMobileMenu();
  initPasswordToggles();
  initFormValidation();
  initFileUpload();
  initAlertAutoDismiss();
});

/* ── Mobile menu ────────────────────────────────────────────── */
function initMobileMenu() {
  const toggle = document.getElementById("mobileToggle");
  const nav = document.querySelector(".nav");
  if (!toggle || !nav) return;

  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    nav.classList.toggle("open");
  });

  // Close when a link is clicked
  nav.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      toggle.classList.remove("open");
      nav.classList.remove("open");
    });
  });
}

/* ── Password visibility toggles ────────────────────────────── */
function initPasswordToggles() {
  document.querySelectorAll(".password-toggle").forEach((btn) => {
    btn.addEventListener("click", () => {
      const input = btn.parentElement.querySelector("input");
      if (!input) return;
      const isPassword = input.type === "password";
      input.type = isPassword ? "text" : "password";
      btn.textContent = isPassword ? "🙈" : "👁";
      btn.setAttribute(
        "aria-label",
        isPassword ? "Hide password" : "Show password"
      );
    });
  });
}

/* ── Form validation ────────────────────────────────────────── */
function initFormValidation() {
  /* --- Signup form --- */
  const signupForm = document.getElementById("signupForm");
  if (signupForm) {
    signupForm.addEventListener("submit", (e) => {
      clearErrors(signupForm);
      let valid = true;

      const fullName = signupForm.querySelector("#full_name");
      const username = signupForm.querySelector("#username");
      const email = signupForm.querySelector("#email");
      const password = signupForm.querySelector("#password");
      const confirm = signupForm.querySelector("#confirm_password");

      if (!fullName.value.trim()) {
        showFieldError(fullName, "Full name is required.");
        valid = false;
      }

      if (!username.value.trim()) {
        showFieldError(username, "Username is required.");
        valid = false;
      } else if (username.value.trim().length < 3) {
        showFieldError(username, "Username must be at least 3 characters.");
        valid = false;
      } else if (!/^[a-zA-Z0-9_]+$/.test(username.value.trim())) {
        showFieldError(
          username,
          "Username can only contain letters, numbers, and underscores."
        );
        valid = false;
      }

      if (!email.value.trim()) {
        showFieldError(email, "Email is required.");
        valid = false;
      } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value.trim())) {
        showFieldError(email, "Please enter a valid email address.");
        valid = false;
      }

      if (!password.value) {
        showFieldError(password, "Password is required.");
        valid = false;
      } else if (password.value.length < 8) {
        showFieldError(password, "Password must be at least 8 characters.");
        valid = false;
      }

      if (password.value !== confirm.value) {
        showFieldError(confirm, "Passwords do not match.");
        valid = false;
      }

      if (!valid) {
        e.preventDefault();
        return;
      }

      // Show loading state
      const btn = signupForm.querySelector('button[type="submit"]');
      btn.classList.add("loading");
      btn.disabled = true;
    });
  }

  /* --- Login form --- */
  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", (e) => {
      clearErrors(loginForm);
      let valid = true;

      const loginId = loginForm.querySelector("#login_id");
      const password = loginForm.querySelector("#password");

      if (!loginId.value.trim()) {
        showFieldError(loginId, "Username or email is required.");
        valid = false;
      }
      if (!password.value) {
        showFieldError(password, "Password is required.");
        valid = false;
      }

      if (!valid) {
        e.preventDefault();
        return;
      }

      const btn = loginForm.querySelector('button[type="submit"]');
      btn.classList.add("loading");
      btn.disabled = true;
    });
  }
}

function showFieldError(input, message) {
  input.classList.add("error");
  const errorEl = input
    .closest(".form-group")
    .querySelector(".form-error");
  if (errorEl) {
    errorEl.textContent = message;
    errorEl.classList.add("visible");
  }
}

function clearErrors(form) {
  form.querySelectorAll(".form-input.error").forEach((el) => {
    el.classList.remove("error");
  });
  form.querySelectorAll(".form-error.visible").forEach((el) => {
    el.classList.remove("visible");
  });
}

/* ── File upload ────────────────────────────────────────────── */
function initFileUpload() {
  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("imageInput");
  const preview = document.getElementById("uploadPreview");
  const previewImg = document.getElementById("previewImage");
  const fileName = document.getElementById("fileName");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeForm = document.getElementById("analyzeForm");

  if (!uploadArea || !fileInput) return;

  // Click to upload
  uploadArea.addEventListener("click", () => fileInput.click());

  // Drag and drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
  });
  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
  });
  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect(fileInput.files[0]);
    }
  });

  // File input change
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  function handleFileSelect(file) {
    if (!file) return;

    // Validate file type
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      alert("Please upload a JPG, PNG, or WebP image.");
      fileInput.value = "";
      return;
    }

    // Validate file size (max 16 MB)
    if (file.size > 16 * 1024 * 1024) {
      alert("File size must be under 16 MB.");
      fileInput.value = "";
      return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      if (previewImg) previewImg.src = e.target.result;
      if (fileName) fileName.textContent = file.name;
      if (preview) preview.classList.add("visible");
      if (analyzeBtn) analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // Form submit loading
  if (analyzeForm) {
    analyzeForm.addEventListener("submit", () => {
      showLoadingOverlay("Analyzing image — this may take a moment…");
    });
  }
}

/* ── Loading overlay ────────────────────────────────────────── */
function showLoadingOverlay(text) {
  let overlay = document.getElementById("loadingOverlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loadingOverlay";
    overlay.className = "loading-overlay";
    overlay.innerHTML = `
      <div class="loading-spinner"></div>
      <div class="loading-text">${text || "Loading…"}</div>
    `;
    document.body.appendChild(overlay);
  } else {
    overlay.querySelector(".loading-text").textContent =
      text || "Loading…";
  }
  // Force reflow before adding class
  void overlay.offsetWidth;
  overlay.classList.add("active");
}

/* ── Alert auto-dismiss ─────────────────────────────────────── */
function initAlertAutoDismiss() {
  document.querySelectorAll(".alert").forEach((alert) => {
    setTimeout(() => {
      alert.style.opacity = "0";
      alert.style.transform = "translateY(-8px)";
      setTimeout(() => alert.remove(), 300);
    }, 6000);
  });
}
