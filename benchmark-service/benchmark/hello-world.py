from aiohttp import web


routes = web.RouteTableDef()


@routes.get('/')
async def handler(request: web.Request) -> web.Response:
    return web.Response(text="Hello world!")


@routes.get('/json')
async def json_handler(request):
    args = await request.json()
    data = {'value': args['key']}
    return web.json_response(data)


@routes.get('/{username}')
async def greet_user(request: web.Request) -> web.Response:
    user = request.match_info.get("username", "")
    return web.Response(text=f"Hello, {user}")


@routes.post('/add_user')
async def add_user(request: web.Request) -> web.Response:
    data = await request.post()
    username = data.get('username')

    return web.Response(text=f"{username} was added")


async def init_app() -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    # app.add_routes([web.get('/', handler)])
    return app


web.run_app(init_app())
