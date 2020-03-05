const sendfile = require('koa-sendfile'); // 下载文件
const fs = require('fs')
const path = require('path')

const iconv = require('iconv-lite');

const {
	exec,
	execFile
} = require('child_process');


let binaryEncoding = 'binary';
let encoding = 'cp936';
let cmd = ''
let res

module.exports = {

	index: async ctx => { // 设置路由  ctx是request和reponse的结合

		const FileName = 'SequenceTemp.txt';
		// const cmdName = 'run.bat';
		const sequencePath = path.join(__dirname, './public/sequences/') + `/${FileName}`;
		// const cmdPath = path.join(__dirname, './public/cmd/') + `/${cmdName}`;
		cmd = 'start \"\" cmd /k \"activate wjy' + '&&' + 
		'python ' + 'public/sequences/out.py' + '&&' + 
		'python ' + 'out.py' + 
		 '\"';

		console.log(cmd);

		fs.writeFile(sequencePath, ctx.request.body.sequence, err => {
			if (err) {
				throw err;
			}
		})

		/*
			fs.writeFile(cmdPath, cmd, err => {
				if(err){
					throw err;
				}
			})
			console.log(cmd);
		*/

		exec(cmd, {
			encoding: binaryEncoding
		}, (error, stdout) => {
			res = 12;
			console.log('stdout1', iconv.decode(new Buffer.from(stdout, binaryEncoding), encoding));
			console.log(ctx.request.body)
		});


		/*
			execFile("./public/cmd/run.bat",null,function(error,stdout,stderr){
			    if(error !==null){
			        console.log("exec error"+error)
			    }
			    else console.log("成功")
			    console.log('stdout: ' + stdout);
			    console.log('stderr: ' + stderr);
			})

		*/

		ctx.body = { // 响应到页面中的数据 ctx.body代表报文的主体
			token: 'abc', // 通过axios会直接把body中的内容返回
			msg: 'ok',
			std: ctx.request.body.sequence
		};
	},

	file: async (ctx, next) => {
		let file = ctx.request.files.file;
		// 创建可读流
		const reader = fs.createReadStream(file.path);
		// 修改文件的名称
		const myDate = new Date();
		// var newFilename = myDate.getTime()+'.'+file.name.split('.')[1];
		const Filename = ctx.request.files.file.name;
		// console.log(ctx.request.files.file.name)
		const targetPath = path.join(__dirname, './public/uploads/') + `/${Filename}`;
		// //创建可写流
		const upStream = fs.createWriteStream(targetPath);
		// // 可读流通过管道写入可写流
		reader.pipe(upStream);
		const url = 'http://' + ctx.headers.host + '/uploads/' + Filename;
		//返回保存的路径
		return ctx.body = {
			code: 200,
			name: Filename,
			data: {
				url: url
			},
			msg: '上传成功'
		};
	},

	downloadParams: async ctx => {
		const name = ctx.params.name;
		console.log(name);
		const path = `./public/uploads/${name}`;
		ctx.attachment(path);
		await sendfile(ctx, path);
	}
}
