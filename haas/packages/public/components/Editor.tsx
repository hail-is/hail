// Based on https://github.com/superRaytin/react-monaco-editor/blob/master/src/editor.js
// but as a PurceComponent + Typescript, and allows us to self-manage
// TODO: understand why import * as monaco works, but not import monaco...
import * as monaco from 'monaco-editor/esm/vs/editor/editor.api';
import { PureComponent } from 'react';

interface MonacoEditor {
  editor: monaco.editor.IStandaloneCodeEditor;
}

export type EditorDidMount = (
  editor: monaco.editor.IStandaloneCodeEditor,
  // TODO: not entirely sure how this works
  monacoEditor: typeof monaco
) => void;

export type EditorWillMount = (
  monacoEditor: typeof monaco
) => void | monaco.editor.IEditorConstructionOptions;

export type ChangeHandler = (
  value: string,
  event: monaco.editor.IModelContentChangedEvent
) => void;

export interface Props {
  width: string | number;
  height: string | number;
  value: string;
  defaultValue: string;
  language: string;
  theme: string;
  options: object;
  editorDidMount: EditorDidMount;
  editorWillMount: EditorWillMount;
  onChange: ChangeHandler;
}

class MonacoEditor extends PureComponent<Props> {
  static defaultProps = {
    width: '100%',
    height: '100%',
    value: null,
    defaultValue: '',
    language: 'python',
    theme: null,
    options: {},
    editorDidMount: () => {},
    editorWillMount: () => {},
    onChange: () => {}
  };

  private _currentValue: string;
  private _containerElement: HTMLDivElement | null;
  private _preventTriggerChangeEvent: boolean;
  private _resizeTimeout?: NodeJS.Timer;

  constructor(props: Props) {
    super(props);

    this._currentValue = props.value;

    if (typeof window === 'undefined') {
      throw new Error('Editor must be used browser-side only');
    }
  }

  componentDidMount = () => {
    window.addEventListener('resize', this.resize);
    this.init();
  };

  init = () => {
    const { value, language, theme, options } = this.props;

    if (this._containerElement) {
      // Before initializing monaco editor
      Object.assign(options, this.editorWillMount());
      console.info('options', options);
      this.editor = monaco.editor.create(this._containerElement, {
        value,
        language,
        ...options
      });
      if (theme) {
        monaco.editor.setTheme(theme);
      }
      // After initializing monaco editor
      this.editorDidMount(this.editor);
    }
  };

  componentDidUpdate = (prevProps: Partial<Props>) => {
    if (this.props.value !== this._currentValue) {
      // Always refer to the latest value
      this._currentValue = this.props.value;
      // Consider the situation of rendering 1+ times before the editor mounted
      if (this.editor) {
        this._preventTriggerChangeEvent = true;
        this.editor.setValue(this._currentValue);
        this._preventTriggerChangeEvent = false;
      }
    }
    if (prevProps.language !== this.props.language) {
      monaco.editor.setModelLanguage(
        this.editor.getModel(),
        this.props.language
      );
    }
    if (prevProps.theme !== this.props.theme) {
      monaco.editor.setTheme(this.props.theme);
    }
    if (
      this.editor &&
      (this.props.width !== prevProps.width ||
        this.props.height !== prevProps.height)
    ) {
      this.editor.layout();
    }
    if (prevProps.options !== this.props.options) {
      this.editor.updateOptions(this.props.options);
    }
  };

  componentWillUnmount = () => {
    window.removeEventListener('resize', this.resize);
    this.destroyMonaco();
  };

  resize = () => {
    clearTimeout(this._resizeTimeout);
    this._resizeTimeout = setTimeout(() => {
      this.editor.layout();
    }, 350);
  };

  editorWillMount = () => {
    const options = this.props.editorWillMount(monaco);
    return options || {};
  };

  editorDidMount = editor => {
    this.props.editorDidMount(editor, monaco);
    editor.onDidChangeModelContent(event => {
      const value = editor.getValue();

      // Always refer to the latest value
      this._currentValue = value;

      // Only invoking when user input changed
      if (!this._preventTriggerChangeEvent) {
        this.props.onChange(value, event);
      }
    });
  };

  destroyMonaco = () => {
    if (typeof this.editor !== 'undefined') {
      this.editor.dispose();
    }
  };

  assignRef = component => {
    this._containerElement = component;
  };

  render() {
    const style = {
      width: this.props.width,
      height: this.props.height
    };

    return (
      <div
        ref={this.assignRef}
        style={style}
        className="react-monaco-editor-container"
      />
    );
  }
}

export default MonacoEditor;
