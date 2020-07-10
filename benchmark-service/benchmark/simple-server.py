from aiohttp import web


async def handler(request):
    return web.Response()


async def hello(request):
    return web.Response(text="Hello, world")


def init_func(argv):
    app = web.Application()
    app.router.add_get("/", index_handler)
    return app


app = web.Application()
app.add_routes([web.get('/', handler),
                web.post])
web.run_app(app)
# routes = web.RouteTableDef()
#
#
# async def handler(request):
#     return web.Response()
