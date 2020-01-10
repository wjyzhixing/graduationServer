const sendfile = require('koa-sendfile'); // 下载文件
const fs = require('fs')
const path = require('path')
const readline = require('readline')

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

		let msg = 'http://localhost:9998/down';
		let resultData = '';
		const FileName = 'SequenceTemp.txt';
		// const cmdName = 'run.bat';
		const sequencePath = path.join(__dirname, './public/sequences/') + `/${FileName}`;
		// const cmdPath = path.join(__dirname, './public/cmd/') + `/${cmdName}`;
		cmd = 'start \"\" cmd /k \"' + 'cd Test' + ' && ' + 
		'python CNN_Testing.py --inputFile ../public/sequences/SequenceTemp.txt --inputModel Model.h5 --Shuffle true' + ' && ' +
		'exit' + '"';
		console.log(cmd);

		await fs.writeFileSync(sequencePath, ctx.request.body.sequence, err => {
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

		await exec(cmd, {
			encoding: binaryEncoding
		}, (error, stdout) => {
			res = 12;
			// console.log('stdout1', iconv.decode(new Buffer.from(stdout, binaryEncoding), encoding));
			
			// 读取结果
			readFileToArr = (fReadName,callback) => {
			    let fRead = fs.createReadStream(fReadName);
			    let objReadline = readline.createInterface({
			        input:fRead
			    });
			    let arr = new Array();
			    objReadline.on('line' , (line) => {
			        arr.push(line);
			        //console.log('line:'+ line);
			    });
			    objReadline.on('close' , () => {
			       // console.log(arr);
			        callback(arr);
			    });
			}
		
			a = 2
			resultData = readFileToArr('Test/Results/Results.txt' , (data) => {
				// console.log(data)
				resultData = data;
				console.log(resultData)
				return resultData;
				// console.log(typeof(data))
			})
			
			console.log(resultData)
			console.log(a)
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

		return ctx.body = { // 响应到页面中的数据 ctx.body代表报文的主体
			token: 'abc', // 通过axios会直接把body中的内容返回
			msg: msg,
			std: '正在预测,请稍后......'
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
		const FileChange = 'SequenceTemp.txt';
		// console.log(ctx.request.files.file.name)
		const targetPath = path.join(__dirname, './public/sequences/') + `/${FileChange}`;
		// //创建可写流
		const upStream = fs.createWriteStream(targetPath);
		// // 可读流通过管道写入可写流
		reader.pipe(upStream);
		const url = 'http://' + ctx.headers.host + '/uploads/' + FileChange;
		
		cmd = 'start \"\" cmd /k \"' + 'cd Test' + ' && ' +
		'python CNN_Testing.py --inputFile ../public/sequences/' + FileChange + ' --inputModel Model.h5 --Shuffle true' + ' && ' + 
		'exit' + '"';
		console.log(cmd);
	
		await exec(cmd, {
			encoding: binaryEncoding
		}, (error, stdout) => {
			res = 12;
			// console.log('stdout1', iconv.decode(new Buffer.from(stdout, binaryEncoding), encoding));
			
			// 读取结果
			readFileToArr = (fReadName,callback) => {
			    let fRead = fs.createReadStream(fReadName);
			    let objReadline = readline.createInterface({
			        input:fRead
			    });
			    let arr = new Array();
			    objReadline.on('line' , (line) => {
			        arr.push(line);
			        //console.log('line:'+ line);
			    });
			    objReadline.on('close' , () => {
			       // console.log(arr);
			        callback(arr);
			    });
			}
		
			a = 2
			resultData = readFileToArr('Test/Results/Results.txt' , (data) => {
				// console.log(data)
				resultData = data;
				console.log(resultData)
				return resultData;
				// console.log(typeof(data))
			})
			
			console.log(resultData)
			console.log(a)
			console.log(ctx.request.body)
		});

		
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
		// const name = ctx.params.name;
		// console.log(name);
		// const path = `./public/uploads/${name}`;
		// ctx.attachment(path);
		// await sendfile(ctx, path);

		const path = `./Test/Results/Results.txt`;
		ctx.attachment(path);
		await sendfile(ctx, path);

	},
	
	
	
	receiveResult: async ctx => {
		const name = 'Test/Results/Results.txt';
		console.log(fs.readFileSync(name,'utf-8'))
		const result = fs.readFileSync(name,'utf-8')
		
		return ctx.body = {
			code: 200,
			name: name,
			msg: result,
		};
	},
	
	down: async ctx => {
		const path = `./Test/Results/Results.txt`;
		ctx.attachment(path);
		await sendfile(ctx, path);
	}
}
