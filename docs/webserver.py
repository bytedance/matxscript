# -*- coding: utf-8 -*-
import os
import tornado.ioloop
import tornado.web


class MyStaticFileHandler(tornado.web.StaticFileHandler):
    def set_extra_headers(self, path):
        # Disable cache
        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')


def make_app():
    base_path = os.path.split(os.path.realpath(__file__))[0]
    path = base_path + os.sep + "build/html"
    return tornado.web.Application([
        (r'/matx4/()', MyStaticFileHandler, {'path': path, 'default_filename': 'index.html'}),
        (r'/matx4/(.*\..*)', MyStaticFileHandler, {'path': path, 'default_filename': 'index.html'}),
    ])


if __name__ == "__main__":
    app = make_app()
    server = tornado.httpserver.HTTPServer(app)
    server.bind(6006)
    server.start(4)
    tornado.ioloop.IOLoop.instance().start()
