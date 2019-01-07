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
    
    jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
    
    typ = hl.dtype(jir.typ().toString())
    value = Env.hail().expr.ir.Interpret.interpretJSON(jir)

    result = {
        'type': str(typ),
        'value': value
    }
    
    info(f'result: {result}')
    
    return flask.jsonify(result)

@app.route('/type/value', methods=['POST'])
def value_type():
    code = flask.request.json

    info(f'value type: {code}')

    jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
    result = jir.typ().toString()

    info(f'result: {result}')

    return flask.jsonify(result)

@app.route('/type/table', methods=['POST'])
def table_type():
    code = flask.request.json

    info(f'table type: {code}')

    jir = Env.hail().expr.ir.IRParser.parse_table_ir(code, {}, {})
    ttyp = hl.ttable._from_java(jir.typ())
    
    result = {'global': str(ttyp.global_type),
              'row': str(ttyp.row_type),
              'row_key': ttyp.row_key}
    
    info(f'result: {result}')

    return flask.jsonify(result)


@app.route('/type/matrix', methods=['POST'])
def matrix_type():
    code = flask.request.json

    info(f'matrix type: {code}')

    jir = Env.hail().expr.ir.IRParser.parse_matrix_ir(code, {}, {})
    mtyp = hl.tmatrix._from_java(jir.typ())
    
    result = {'global': str(mtyp.global_type),
              'col': str(mtyp.col_type),
              'col_key': mtyp.col_key,
              'row': str(mtyp.row_type),
              'row_key': mtyp.row_key,
              'entry': str(mtyp.entry_type)}
    
    info(f'result: {result}')

    return flask.jsonify(result)

app.run(threaded=False, host='0.0.0.0')
