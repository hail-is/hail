import hail as hl

from hail.utils.java import Env, info

import logging
import flask

hl.init()

app = flask.Flask('hail-apiserver')

@app.route('/execute', methods=['POST'])
def execute():
    code = flask.request.json
    
    info(f'execute: {code}')
    
    jir = Env.hail().expr.Parser.parse_value_ir(code, {}, {})
    
    typ = hl.HailType._from_java(jir.typ())
    value = Env.hail().expr.ir.Interpret.interpretPyIR(code, {}, {})

    result = {
        'type': str(typ),
        'value': value
    }
    
    info(f'result: {result}')
    
    return flask.jsonify(result)

app.run(threaded=False, host='0.0.0.0')
