let mongoose = require("mongoose");
let config = require("./config.js");

module.exports = function(){
	let db = mongoose.connect(config.mongodb,{useNewUrlParser:true},function(err){
		if(err){
			console.log("Connect error:" + err);
		}
		else{
			console.log("connect success");
		}
	});
	
	require("../models/user.server.model.js");
	return db;
}