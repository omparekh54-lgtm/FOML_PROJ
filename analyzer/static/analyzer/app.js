(() => {
  const form=document.getElementById("analyzer-form");
  const input=document.getElementById("file-input");
  const preview=document.getElementById("dz-preview");
  const empty=document.getElementById("dz-empty");
  const result=document.getElementById("result");
  input.addEventListener("change",()=>{const f=input.files[0]; if(!f)return; preview.src=URL.createObjectURL(f); preview.hidden=false; empty.hidden=true;});
  form.addEventListener("submit",async(e)=>{
    e.preventDefault(); if(!input.files[0])return;
    result.textContent="Analyzing…";
    const fd=new FormData(form);
    const resp=await fetch("/api/analyze/",{method:"POST",body:fd,headers:{"X-CSRFToken":document.querySelector('meta[name="csrf-token"]').content}});
    const data=await resp.json();
    result.innerHTML="";
    const box=document.createElement("div"); box.className="card";
    if(data.status!=="ok"){box.innerHTML="<h3>"+(data.status||"Error")+"</h3><p>"+(data.message||"Analysis unavailable.")+"</p>";}
    else{
      const conf=typeof data.abnormality_confidence==="number" ? Math.round(data.abnormality_confidence*100)+"%" : "—";
      box.innerHTML="<h3>"+(data.body_part||"Unknown body part")+"</h3><strong>"+(data.abnormality_label||"unknown")+"</strong><p>Confidence: "+conf+"</p><p>"+(data.explanation||"No explanation available.")+"</p>";
    }
    result.appendChild(box);
  });
})();