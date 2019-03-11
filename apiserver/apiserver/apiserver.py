import hail as hl

from hail.utils import FatalError
from hail.utils.java import Env, info, scala_object

import os
import logging
import flask
import json

master = os.environ.get('HAIL_APISERVER_SPARK_MASTER')
hl.init(master=master, min_block_size=0)

app = flask.Flask('hail-apiserver')

@app.route('/healthcheck')
def healthcheck():
    return '', 200

@app.route('/execute', methods=['POST'])
def execute():
    code = flask.request.json
    info(f'execute: {code}')
    try:
        jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
        typ = hl.dtype(jir.typ().toString())
        value = Env.hail().backend.spark.SparkBackend.executeJSON(jir)
        result = {
            'type': str(typ),
            'value': value
        }
        info(f'result: {result}')
        return flask.jsonify(result)
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400


@app.route('/type/value', methods=['POST'])
def value_type():
    code = flask.request.json
    info(f'value type: {code}')
    try:
        jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
        result = jir.typ().toString()
        info(f'result: {result}')
        return flask.jsonify(result)
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400
    

@app.route('/type/table', methods=['POST'])
def table_type():
    code = flask.request.json
    info(f'table type: {code}')
    try:
        jir = Env.hail().expr.ir.IRParser.parse_table_ir(code, {}, {})
        ttyp = hl.ttable._from_java(jir.typ())
        result = {'global': str(ttyp.global_type),
                  'row': str(ttyp.row_type),
                  'row_key': ttyp.row_key}
        info(f'result: {result}')
        return flask.jsonify(result)
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400


@app.route('/type/matrix', methods=['POST'])
def matrix_type():
    code = flask.request.json
    info(f'matrix type: {code}')
    try:
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
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400


@app.route('/type/blockmatrix', methods=['POST'])
def blockmatrix_type():
    code = flask.request.json
    info(f'blockmatrix type: {code}')
    try:
        jir = Env.hail().expr.ir.IRParser.parse_blockmatrix_ir(code, {}, {})
        bmtyp = hl.tblockmatrix._from_java(jir.typ())
        result = {'element_type': str(bmtyp.element_type),
                  'shape': bmtyp.shape,
                  'is_row_vector': bmtyp.is_row_vector,
                  'block_size': bmtyp.block_size}
        info(f'result: {result}')
        return flask.jsonify(result)
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400


@app.route('/references/create', methods=['POST'])
def create_reference():
    try:
        config = flask.request.json
        hl.ReferenceGenome._from_config(config)
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/create/fasta', methods=['POST'])
def create_reference_from_fasta():
    try:
        data = flask.request.json
        hl.ReferenceGenome.from_fasta_file(
            data['name'],
            data['fasta_file'],
            data['index_file'],
            data['x_contigs'],
            data['y_contigs'],
            data['mt_contigs'],
            data['par'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/delete', methods=['DELETE'])
def delete_reference():
    try:
        data = flask.request.json
        Env.hail().variant.ReferenceGenome.removeReference(data['name'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/get', methods=['GET'])
def get_reference():
    try:
        data = flask.request.json
        return flask.jsonify(
            json.loads(Env.hail().variant.ReferenceGenome.getReference(data['name']).toJSONString()))
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/sequence/set', methods=['POST'])
def reference_add_sequence():
    try:
        data = flask.request.json
        scala_object(Env.hail().variant, 'ReferenceGenome').addSequence(data['name'], data['fasta_file'], data['index_file'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/sequence/delete', methods=['DELETE'])
def reference_remove_sequence():
    try:
        data = flask.request.json
        scala_object(Env.hail().variant, 'ReferenceGenome').removeSequence(data['name'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/liftover/add', methods=['POST'])
def reference_add_liftover():
    try:
        data = flask.request.json
        Env.hail().variant.ReferenceGenome.referenceAddLiftover(data['name'], data['chain_file'], data['dest_reference_genome'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/references/liftover/remove', methods=['DELETE'])
def reference_remove_liftover():
    try:
        data = flask.request.json
        Env.hail().variant.ReferenceGenome.referenceRemoveLiftover(data['name'], data['dest_reference_genome'])
        return '', 204
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

@app.route('/parse-vcf-metadata', methods=['POST'])
def parse_vcf_metadata():
    try:
        data = flask.request.json
        return flask.jsonify(
            json.loads(Env.hc()._jhc.pyParseVCFMetadataJSON(data['path'])))
    except FatalError as e:
        return flask.jsonify({
            'message': e.args[0]
        }), 400

app.run(threaded=False, host='0.0.0.0')
