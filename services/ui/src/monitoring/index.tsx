import { createRoot } from 'react-dom/client';

function HelloReact() {
  return <p style={{ fontFamily: 'sans-serif', padding: '1rem' }}>Hello React (monitoring)</p>;
}

const el = document.getElementById('monitoring-react-root');
if (el) createRoot(el).render(<HelloReact />);
