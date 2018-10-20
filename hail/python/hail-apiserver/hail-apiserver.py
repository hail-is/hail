import hail as hl

from hail.utils.java import Env

import flask

hl.init()

app = flask.Flask('hail-apiserver')

@app.route('/execute', methods=['POST'])
def execute():
    code = flask.request.json
    
    print('code', code)
    
    jir = Env.hail().expr.Parser.parse_value_ir(code, {}, {})
    
    typ = hl.HailType._from_java(jir.typ())
    result = Env.hail().expr.ir.Interpret.interpretPyIR(code, {}, {})
    
    return flask.jsonify({
        'type': str(typ),
        'value': result
    })

app.run(threaded=False, host='0.0.0.0')
