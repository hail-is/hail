import { createRoot } from 'react-dom/client';

function HelloReact() {
  return <p style={{ fontFamily: 'sans-serif', padding: '1rem' }}>Hello React (batch-driver)</p>;
}

const el = document.getElementById('batch-driver-react-root');
if (el) createRoot(el).render(<HelloReact />);
