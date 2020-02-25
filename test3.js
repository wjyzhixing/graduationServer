var child_process = require('child_process');
// let binaryEncoding = 'binary';
// let encoding = 'utf-8';
var spawnObj = child_process.spawn('python', ['Test/CNN_Testing.py','--inputFile','../public/sequences/SequenceTemp.txt','--inputModel','Model.h5']);
spawnObj.stdout.on('data', function(chunk) {
    console.log(chunk.toString());
	
});
spawnObj.stderr.on('data', (data) => {
  console.log(data);
});
spawnObj.on('close', function(code) {
    console.log('close code : ' + code);
});
spawnObj.on('exit', (code) => {
    console.log('exit code : ' + code);
    // fs.close(fd, function(err) {
    //     if(err) {
    //         console.error(err);
    //     }
    // });
});