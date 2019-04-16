import asyncio
import concurrent
import json
import os
import uvloop
from aiohttp import web

import jwt
import hail as hl
from hail.utils import FatalError
from hail.utils.java import Env, info, scala_object
import hailjwt as hj

uvloop.install()

master = os.environ.get('HAIL_APISERVER_SPARK_MASTER')
hl.init(master=master, min_block_size=0)

app = web.Application()
routes = web.RouteTableDef()


with open(os.environ.get('HAIL_JWT_SECRET_KEY_FILE') or '/jwt-secret/secret-key') as f:
    jwtclient = hj.JWTClient(f.read())


def authenticated_users_only(fun):
    def wrapped(request, *args, **kwargs):
        encoded_token = request.cookies.get('user')
        if encoded_token is not None:
            try:
                userdata = jwtclient.decode(encoded_token)
                if 'userdata' in fun.__code__.co_varnames:
                    return fun(request, *args, userdata=userdata, **kwargs)
                return fun(request, *args, **kwargs)
            except jwt.exceptions.DecodeError:
                pass
        raise web.HTTPForbidden()
    wrapped.__name__ = fun.__name__
    return wrapped


def status_response(status):
    return web.Response(status=status)


executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)


async def run(f, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, f, *args)


@routes.get('/healthcheck')
async def healthcheck(request):
    del request
    return status_response(200)


def blocking_execute(code):
    jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
    typ = hl.dtype(jir.typ().toString())
    result = Env.hail().backend.spark.SparkBackend.executeJSON(jir)
    return {
        'type': str(typ),
        'result': result
    }


@routes.post('/execute')
@authenticated_users_only
async def execute(request):
    code = await request.json()
    info(f'execute: {code}')
    try:
        result = await run(blocking_execute, code)
        info(f'result: {result}')
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_value_type(code):
    jir = Env.hail().expr.ir.IRParser.parse_value_ir(code, {}, {})
    return jir.typ().toString()


@routes.post('/type/value')
@authenticated_users_only
async def value_type(request):
    code = await request.json()
    info(f'value type: {code}')
    try:
        result = await run(blocking_value_type, code)
        info(f'result: {result}')
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_table_type(code):
    jir = Env.hail().expr.ir.IRParser.parse_table_ir(code, {}, {})
    ttyp = hl.ttable._from_java(jir.typ())
    return {
        'global': str(ttyp.global_type),
        'row': str(ttyp.row_type),
        'row_key': ttyp.row_key
    }


@routes.post('/type/table')
@authenticated_users_only
async def table_type(request):
    code = await request.json()
    info(f'table type: {code}')
    try:
        result = await run(blocking_table_type, code)
        info(f'result: {result}')
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_matrix_type(code):
    jir = Env.hail().expr.ir.IRParser.parse_matrix_ir(code, {}, {})
    mtyp = hl.tmatrix._from_java(jir.typ())
    return {
        'global': str(mtyp.global_type),
        'col': str(mtyp.col_type),
        'col_key': mtyp.col_key,
        'row': str(mtyp.row_type),
        'row_key': mtyp.row_key,
        'entry': str(mtyp.entry_type)
    }


@routes.post('/type/matrix')
@authenticated_users_only
async def matrix_type(request):
    code = await request.json()
    info(f'matrix type: {code}')
    try:
        result = await run(blocking_matrix_type, code)
        info(f'result: {result}')
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_blockmatrix_type(code):
    jir = Env.hail().expr.ir.IRParser.parse_blockmatrix_ir(code, {}, {})
    bmtyp = hl.tblockmatrix._from_java(jir.typ())
    return {
        'element_type': str(bmtyp.element_type),
        'shape': bmtyp.shape,
        'is_row_vector': bmtyp.is_row_vector,
        'block_size': bmtyp.block_size
    }


@routes.post('/type/blockmatrix')
@authenticated_users_only
async def blockmatrix_type(request):
    code = await request.json()
    info(f'blockmatrix type: {code}')
    try:
        result = await run(blocking_blockmatrix_type, code)
        info(f'result: {result}')
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


@routes.post('/references/create')
@authenticated_users_only
async def create_reference(request):
    try:
        config = await request.json()
        hl.ReferenceGenome._from_config(config)
        return status_response(204)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


@routes.post('/references/create/fasta')
@authenticated_users_only
async def create_reference_from_fasta(request):
    try:
        data = await request.json()
        hl.ReferenceGenome.from_fasta_file(
            data['name'],
            data['fasta_file'],
            data['index_file'],
            data['x_contigs'],
            data['y_contigs'],
            data['mt_contigs'],
            data['par'])
        return status_response(204)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_get_reference(data):
    return json.loads(
        Env.hail().variant.ReferenceGenome.getReference(
            data['name']
        ).toJSONString())


@routes.get('/references/get')
@authenticated_users_only
async def get_reference(request):
    try:
        data = await request.json()
        result = await run(blocking_get_reference, data)
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_reference_add_sequence(data):
    scala_object(Env.hail().variant, 'ReferenceGenome').addSequence(
        data['name'], data['fasta_file'], data['index_file'])


@routes.post('/references/sequence/set')
@authenticated_users_only
async def reference_add_sequence(request):
    try:
        data = await request.json()
        await run(blocking_reference_add_sequence, data)
        return status_response(204)
    except FatalError as e:
        return web.json_response({'message': e.args[0]}, status=400)


def blocking_reference_remove_sequence(data):
    scala_object(
        Env.hail().variant, 'ReferenceGenome').removeSequence(data['name'])


@routes.delete('/references/sequence/delete')
@authenticated_users_only
async def reference_remove_sequence(request):
    try:
        data = await request.json()
        await run(blocking_reference_remove_sequence, data)
        return status_response(204)
    except FatalError as e:
        return web.json_response({
            'message': e.args[0]
        }, status=400)


def blocking_reference_add_liftover(data):
    Env.hail().variant.ReferenceGenome.referenceAddLiftover(
        data['name'],
        data['chain_file'],
        data['dest_reference_genome'])


@routes.post('/references/liftover/add')
@authenticated_users_only
async def reference_add_liftover(request):
    try:
        data = await request.json()
        await run(blocking_reference_add_liftover, data)
        return status_response(204)
    except FatalError as e:
        return web.json_response({'message': e.args[0]}, status=400)


def blocking_reference_remove_liftover(data):
    Env.hail().variant.ReferenceGenome.referenceRemoveLiftover(
        data['name'],
        data['dest_reference_genome'])


@routes.delete('/references/liftover/remove')
@authenticated_users_only
async def reference_remove_liftover(request):
    try:
        data = await request.json()
        await run(blocking_reference_remove_liftover, data)
        return status_response(204)
    except FatalError as e:
        return web.json_response({'message': e.args[0]}, status=400)


def blocking_parse_vcf_metadata(data):
    return json.loads(Env.hc()._jhc.pyParseVCFMetadataJSON(data['path']))  # pylint: disable=no-member


@routes.post('/parse-vcf-metadata')
@authenticated_users_only
async def parse_vcf_metadata(request):
    try:
        data = await request.json()
        result = await run(blocking_parse_vcf_metadata, data)
        return web.json_response(result)
    except FatalError as e:
        return web.json_response({'message': e.args[0]}, status=400)


app.add_routes(routes)
web.run_app(app, host='0.0.0.0', port=5000)
