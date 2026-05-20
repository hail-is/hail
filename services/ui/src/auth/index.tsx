import { createRoot } from 'react-dom/client';

function HelloReact() {
  return <p style={{ fontFamily: 'sans-serif', padding: '1rem' }}>Hello React (auth)</p>;
}

const el = document.getElementById('auth-react-root');
if (el) createRoot(el).render(<HelloReact />);
