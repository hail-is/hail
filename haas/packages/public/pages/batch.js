import { PureComponent } from 'react';
import dynamic from 'next/dynamic';

const MonacoEditor = dynamic(() => import('react-monaco-editor'), {
  ssr: false
});

class App extends PureComponent {
  constructor(props) {
    super(props);
    this.state = {
      code: '// type your code...'
    };
  }
  editorDidMount = (editor, monaco) => {
    console.log('editorDidMount', editor);
    editor.focus();
  };
  onChange = (newValue, e) => {
    console.log('onChange', newValue, e);
  };
  render() {
    const code = this.state.code;
    const options = {
      selectOnLineNumbers: true
    };

    return (
      <MonacoEditor
        width="800"
        height="800"
        language="python"
        value={code}
        options={options}
        onChange={this.onChange}
        editorDidMount={this.editorDidMount}
      />
    );

    // return <div>Loading</div>;
  }
}

export default App;
