const { exec } = require('child_process');

exec('python out.py', { encoding: 'UTF-8' }, (error, stdout) => {
    console.log('stdout1', stdout);
});