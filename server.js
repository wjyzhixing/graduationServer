const Koa = require('koa2'); // 加载koa2模块
let app = new Koa(); // 实例化
const cors = require('koa-cors'); // 跨域
const koaBody = require('koa-body'); // 解析request

const Router = require('./router')

app.use(cors())

app.use(koaBody({
	multipart: true,
	formidable: {
		maxFileSize: 200 * 1024 * 1024 // 设置上传文件大小最大限制，默认2M
	}
}))

Router(app)

app.listen(9998, () => { // 对服务的监听
	console.log("服务启动成功端口号为9998")
})
