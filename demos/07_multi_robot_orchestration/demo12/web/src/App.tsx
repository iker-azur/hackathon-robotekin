import React, { useEffect, useState } from "react";

export default function App() {
  const [state, setState] = useState<any>(null);
  useEffect(() => {
    const ws = new WebSocket("ws://127.0.0.1:8000/ws");
    ws.onmessage = (e) => setState(JSON.parse(e.data));
    return () => ws.close();
  }, []);
  return (
    <div style={{ fontFamily: "sans-serif", padding: 16 }}>
      <h1>Demo 12 — Multi-Robot Orchestration</h1>
      {state ? (
        <pre>{JSON.stringify(state, null, 2)}</pre>
      ) : (
        <p>Connecting…</p>
      )}
    </div>
  );
}
