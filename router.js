const router = require('koa-router')();
const HomeController = require('./home')

module.exports = (app) => {

	router.post('/', HomeController.index);
	router.post('/file', HomeController.file);
	router.get('/download/:name', HomeController.downloadParams);
	router.post('/receiveResult', HomeController.receiveResult);
	router.get('/down', HomeController.down);

	app.use(router.routes()); //把router对象的routes挂载到app上不然找不到，使用这个中间件
}
