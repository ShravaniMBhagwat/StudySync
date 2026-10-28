'use strict';

/* ================================================================
   script.js — StudySync browser interactions
   ================================================================
   1. Mobile navigation toggle
   2. Upload zone: drag-and-drop + file name display + has-file state
   3. Form submission: loading overlay
   4. Flash auto-dismiss
   5. Confidence bar animation
   6. Analytics tab switcher
   7. Keyboard accessibility helpers
   ================================================================ */

/* ── 1. Mobile navigation ────────────────────────────────────────── */
(function initNav() {
  const toggle = document.getElementById('navToggle');
  const menu   = document.getElementById('navLinks');
  if (!toggle || !menu) return;

  toggle.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    toggle.setAttribute('aria-expanded', String(open));
  });

  // Close menu on any link click
  menu.querySelectorAll('a').forEach(a =>
    a.addEventListener('click', () => {
      menu.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    })
  );

  // Close on Escape
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && menu.classList.contains('open')) {
      menu.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
      toggle.focus();
    }
  });
})();

/* ── 2. Upload zone helpers ──────────────────────────────────────── */
function initUploadZone({ zoneId, inputId, labelId, clearId = null, multiple = false }) {
  const zone  = document.getElementById(zoneId);
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (!zone || !input || !label) return;

  const clearBtn = clearId ? document.getElementById(clearId) : null;

  function setFiles(files) {
    if (!files || files.length === 0) {
      label.textContent = '';
      zone.classList.remove('has-file');
      if (clearBtn) clearBtn.style.display = 'none';
      return;
    }
    if (multiple && files.length > 1) {
      label.textContent = `✓ ${files.length} files selected`;
    } else {
      label.textContent = `✓ ${files[0].name}`;
    }
    zone.classList.add('has-file');
    if (clearBtn) clearBtn.style.display = 'inline-block';
  }

  // Drag events
  ['dragenter', 'dragover'].forEach(evt => {
    zone.addEventListener(evt, e => {
      e.preventDefault();
      zone.classList.add('dragging');
    });
  });

  ['dragleave', 'drop'].forEach(evt => {
    zone.addEventListener(evt, () => zone.classList.remove('dragging'));
  });

  zone.addEventListener('drop', e => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      try { input.files = e.dataTransfer.files; } catch(_) {}
      setFiles(e.dataTransfer.files);
    }
  });

  input.addEventListener('change', () => setFiles(input.files));

  // Clear button
  if (clearBtn) {
    clearBtn.addEventListener('click', e => {
      e.preventDefault();
      e.stopPropagation();
      input.value = '';
      setFiles(null);
    });
  }
}

// Initialise all upload zones with multiple file support
initUploadZone({
  zoneId: 'noteUploadZone',
  inputId: 'noteFileInput',
  labelId: 'noteFilename',
  clearId: 'noteClear',
  multiple: true
});

initUploadZone({
  zoneId: 'syllabusZone',
  inputId: 'syllabusInput',
  labelId: document.getElementById('syllabusFilename') ? 'syllabusFilename' : 'syllabusName',
  clearId: 'syllabusClear',
  multiple: true
});

initUploadZone({
  zoneId: 'pyqZone',
  inputId: 'pyqInput',
  labelId: document.getElementById('pyqFilenames') ? 'pyqFilenames' : 'pyqNames',
  clearId: 'pyqClear',
  multiple: true
});

/* ── 3. Form submission: show loading overlay ───────────────────── */
(function initFormLoading() {
  const overlay = document.getElementById('loadingOverlay');
  const loadTxt = document.getElementById('loadingText');
  if (!overlay) return;

  document.querySelectorAll('form[data-loading]').forEach(form => {
    form.addEventListener('submit', () => {
      if (form.checkValidity()) {
        const msg = form.dataset.loading || 'Processing your files…';
        if (loadTxt) loadTxt.textContent = msg;
        overlay.classList.add('active');
      }
    });
  });
})();

/* ── 4. Flash auto-dismiss ───────────────────────────────────────── */
(function initFlashDismiss() {
  document.querySelectorAll('.flash').forEach(flash => {
    setTimeout(() => {
      flash.style.transition = 'opacity .4s ease, max-height .4s ease';
      flash.style.opacity    = '0';
      flash.style.maxHeight  = '0';
      flash.style.overflow   = 'hidden';
      setTimeout(() => flash.remove(), 420);
    }, 6000);
  });
})();

/* ── 5. Confidence bar animation ─────────────────────────────────── */
(function initConfBars() {
  document.querySelectorAll('.conf-fill').forEach(bar => {
    const target = bar.style.width || bar.getAttribute('data-width') || '0%';
    bar.style.width = '0';
    requestAnimationFrame(() => requestAnimationFrame(() => {
      bar.style.width = target;
    }));
  });
})();

/* ── 6. Analytics tab switcher ──────────────────────────────────── */
(function initTabs() {
  document.querySelectorAll('.analytics-tabs').forEach(tabBar => {
    const tabs   = tabBar.querySelectorAll('.analytics-tab');
    const panels = tabBar.closest('section')
                     ? tabBar.closest('section').querySelectorAll('.analytics-panel')
                     : document.querySelectorAll('.analytics-panel');

    tabs.forEach((tab, i) => {
      tab.addEventListener('click', () => {
        tabs.forEach(t   => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
        panels.forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        if (panels[i]) panels[i].classList.add('active');
      });

      tab.addEventListener('keydown', e => {
        const count = tabs.length;
        let idx = i;
        if (e.key === 'ArrowRight') idx = (i + 1) % count;
        if (e.key === 'ArrowLeft')  idx = (i - 1 + count) % count;
        if (idx !== i) { e.preventDefault(); tabs[idx].click(); tabs[idx].focus(); }
      });
    });
  });
})();

/* ── 7. Keyboard: skip-to-content shortcut ─────────────────────── */
(function initSkipLink() {
  const main = document.getElementById('main-content');
  if (!main) return;
  document.addEventListener('keydown', e => {
    if (e.altKey && e.key === 'm') { e.preventDefault(); main.focus(); }
  });
})();
