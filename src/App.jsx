// src/App.jsx (simplified)
import React, {useState} from "react";
export default function App(){
  const [file, setFile] = useState(null);
  const [pred, setPred] = useState(null);
  const [chat, setChat] = useState([]);
  const [msg, setMsg] = useState("");

  const upload = async () => {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("https://YOUR_BACKEND/predict", {method:"POST", body: form});
    const json = await res.json();
    setPred(json);
  };

  const sendMessage = async () => {
    setChat(c => [...c, {from:"user", text:msg}]);
    const res = await fetch("https://YOUR_BACKEND/chat", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({message: msg, context:{/* last sku if any */}})});
    const j = await res.json();
    setChat(c => [...c, {from:"bot", text: j.answer}]);
    setMsg("");
  };

  return (
    <div className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Visual Product Classifier + QA Bot</h1>
      <div className="mb-4">
        <input type="file" onChange={e => setFile(e.target.files[0])} />
        <button disabled={!file} onClick={upload} className="ml-2 px-3 py-1 bg-slate-700 text-white rounded">Upload</button>
      </div>
      {pred && (
        <div className="border p-4 mb-4 rounded">
          <div className="font-semibold">Top-1: {pred.top1_label}</div>
          <div className="text-sm text-gray-600">{pred.short_description}</div>
        </div>
      )}
      <div className="border p-4 rounded">
        <div className="h-48 overflow-auto mb-2">
          {chat.map((c,i)=> <div key={i} className={c.from==="user"?"text-right":"text-left"}>{c.text}</div>)}
        </div>
        <div className="flex gap-2">
          <input value={msg} onChange={e=>setMsg(e.target.value)} className="flex-1 border p-2 rounded" />
          <button onClick={sendMessage} className="px-4 py-2 bg-blue-600 text-white rounded">Send</button>
        </div>
      </div>
    </div>
  );
}