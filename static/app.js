// ===== Drawer =====
const drawer = document.getElementById('drawer');
const scrim = document.getElementById('scrim');
const btnMenu = document.getElementById('btnMenu');
const btnClose = document.getElementById('btnClose');

function openDrawer(){drawer.classList.add('open'); scrim.classList.add('show');}
function closeDrawer(){drawer.classList.remove('open'); scrim.classList.remove('show');}
btnMenu?.addEventListener('click', openDrawer);
btnClose?.addEventListener('click', closeDrawer);
scrim?.addEventListener('click', closeDrawer);

// ===== Dark mode & Nepali font persistence =====
const toggleDark = document.getElementById('toggleDark');
const toggleNepaliFont = document.getElementById('toggleNepaliFont');

function applyTheme(){
  const isDark = localStorage.getItem('theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'dark':'light');
  if (toggleDark) toggleDark.checked = isDark;
}
function applyFont(){
  const np = localStorage.getItem('np-font') === '1';
  document.body.classList.toggle('lang-np', np);
  if (toggleNepaliFont) toggleNepaliFont.checked = np;
}
applyTheme();
applyFont();

toggleDark?.addEventListener('change', () => {
  localStorage.setItem('theme', toggleDark.checked ? 'dark':'light');
  applyTheme();
});
toggleNepaliFont?.addEventListener('change', () => {
  localStorage.setItem('np-font', toggleNepaliFont.checked ? '1':'0');
  applyFont();
});

// ===== Language selection mirrored with drawer radios =====
const langSelect = document.getElementById('langSelect');
const radios = document.querySelectorAll('input[name="lang"]');
radios.forEach(r => r.addEventListener('change', () => {
  langSelect.value = document.querySelector('input[name="lang"]:checked').value;
}));
langSelect?.addEventListener('change', () => {
  const val = langSelect.value;
  document.querySelectorAll('input[name="lang"]').forEach(r => r.checked = (r.value === val));
});

// ===== Loading overlay on submit =====
const form = document.getElementById('analyzeForm');
const loading = document.getElementById('loading');
form?.addEventListener('submit', () => { loading.classList.remove('hidden'); });

// ===== Feedback modal =====
const btnReport = document.getElementById('btnReport');
const dlg = document.getElementById('dlgFeedback');
const fbSend = document.getElementById('fbSend');
const fbText = document.getElementById('fbText');
btnReport?.addEventListener('click', () => { dlg.showModal(); });
fbSend?.addEventListener('click', async (e) => {
  e.preventDefault();
  const message = fbText.value.trim();
  if(!message) { dlg.close(); return; }
  try{
    await fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message})});
  }catch(_){}
  fbText.value=''; dlg.close();
});

// ===== Font size controls (- / +) with persistence =====
const base = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
const fzMinus = document.getElementById('fzMinus');
const fzPlus  = document.getElementById('fzPlus');
const fzInd   = document.getElementById('fzIndicator');

function applyFz(){
  const scale = parseFloat(localStorage.getItem('ui-fz-scale') || '1');
  document.documentElement.style.fontSize = (base * scale) + 'px';
  if (fzInd) fzInd.style.transform = `scale(${scale})`;
}
applyFz();

fzMinus?.addEventListener('click', ()=>{
  const s = Math.max(0.85, (parseFloat(localStorage.getItem('ui-fz-scale')||'1') - 0.05));
  localStorage.setItem('ui-fz-scale', s.toFixed(2)); applyFz();
});
fzPlus?.addEventListener('click', ()=>{
  const s = Math.min(1.30, (parseFloat(localStorage.getItem('ui-fz-scale')||'1') + 0.05));
  localStorage.setItem('ui-fz-scale', s.toFixed(2)); applyFz();
});

// ===== Image Lightbox (enlarge on click) =====
const lb = document.getElementById('lightbox');
const lbImg = document.getElementById('lbImg');
const lbClose = document.getElementById('lbClose');

function openLightbox(src){
  lbImg.src = src; lb.classList.remove('hidden');
}
function closeLightbox(){
  lb.classList.add('hidden'); lbImg.src = '';
}
document.querySelectorAll('img[data-enlarge]')?.forEach(img=>{
  img.addEventListener('click', ()=> openLightbox(img.src));
});
lbClose?.addEventListener('click', closeLightbox);
lb?.addEventListener('click', (e)=>{ if(e.target === lb) closeLightbox(); });
document.addEventListener('keydown', (e)=>{ if(e.key === 'Escape') closeLightbox(); });
