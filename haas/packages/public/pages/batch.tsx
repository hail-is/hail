import { PureComponent } from 'react';
import dynamic from 'next/dynamic';
import Prism from '../lib/prism';
// import 'styles/prism.css';
import 'styles/batch.scss';

const MonacoEditor = dynamic(import('../components/Editor') as any, {
  ssr: false
});

const whitespaceOnly = /^\s+$/g;

interface SelectElem extends React.FormEvent {
  currentTarget: HTMLSelectElement;
}

declare type State = {
  code: string;
  savedCode: string[];
  language: 'python' | 'json' | 'javascript';
};

class App extends PureComponent {
  state: State = {
    code: '',
    savedCode: [],
    language: 'python'
  };

  //https://stackoverflow.com/questions/4446987/overriding-controls-save-functionality-in-browser
  onSaveCommand = (e: KeyboardEvent): EventListener => {
    if (
      (e.keyCode == 83 || e.keyCode == 13) &&
      (navigator.platform.match('Mac') ? e.metaKey : e.ctrlKey)
    ) {
      e.preventDefault();
      console.info('stuff', this.state.code);
      if (!this.state.code) {
        console.info('nothing');
        return;
      }

      this.setState((prevState: State) => {
        prevState.savedCode.push(prevState.code);

        return {
          savedCode: prevState.savedCode,
          code: ''
        };
      });

      console.info(this.state.savedCode);
      Prism.highlightAll();
    }
  };

  componentDidMount = () => {
    document.addEventListener('keydown', this.onSaveCommand, false);
    Prism.highlightAll();
  };

  componentWillUnmount = () => {
    document.removeEventListener('keydown', this.onSaveCommand, false);
  };

  editorDidMount = (editor, monaco) => {
    console.log('editorDidMount', editor);
    editor.focus();
  };

  onChange = (newValue, e) => {
    if (!newValue.match(whitespaceOnly)) {
      this.setState({
        code: newValue
      });
    }
  };

  onDragStart = (e: React.DragEvent, code: string) => {
    console.info('e', code);
    e.dataTransfer.setData('text', code);
  };

  onLanguageSelect = (e: SelectElem) => {
    this.setState({
      // Current target, to get val of bound-to element
      language: e.currentTarget.value
    });
  };

  render() {
    return (
      <span
        style={{
          display: 'flex',
          flexDirection: 'row',
          maxWidth: '100%'
        }}
      >
        <span
          style={{
            display: 'flex',
            flexDirection: 'column',
            maxWidth: '100%'
          }}
        >
          <span
            style={{
              display: 'flex',
              flexDirection: 'row',
              maxWidth: '100%'
            }}
          >
            <select style={{ marginBottom: '15px' }}>
              <option value="alpine">Alpine</option>>
            </select>

            <select
              style={{ marginBottom: '15px', marginLeft: '15px' }}
              onChange={this.onLanguageSelect}
            >
              <option value="python">Python</option>
              <option value="json">JSON</option>
              <option value="typescript">Typescript</option>
              <option value="javascript">Javascript</option>
            </select>
          </span>

          <MonacoEditor
            width="67vw"
            height="90vh"
            language={this.state.language}
            theme="vs-dark"
            value={this.state.code}
            onChange={this.onChange}
            onSave={this.onSaveCommand}
            editorDidMount={this.editorDidMount}
            onDrop={console.log('dropped')}
          />
        </span>
        <div style={{ width: '33vw', marginLeft: 15 }} id="batch">
          {this.state.savedCode.map(codeBlock => (
            <pre
              className="language-python"
              draggable
              onDragStart={e => this.onDragStart(e, codeBlock)}
              onClick={() => this.setState({ code: codeBlock })}
            >
              <code>{codeBlock}</code>
            </pre>
          ))}
        </div>
      </span>
    );

    // return <div>Loading</div>;
  }
}

export default App;
