import React from 'react';
import { Switch, Route } from 'react-router-dom';
import BatchPage from './pages/BatchPage';
import BatchesPage from './pages/BatchesPage';

import '@hail/common/hail.css';

export default function App() {
  return (
    <div>
      <header className="App">
        <Switch>
          <Route exact path="/batches/:id" component={BatchPage} />
          <Route exact path="/" component={BatchesPage} />
          <Route exact path="*" component={BadRoute} />
        </Switch>
      </header>
    </div>
  );
}

function BadRoute() {
  return <p>Uh oh! That page does not exist...</p>;
}
