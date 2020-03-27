let mongoose = require("mongoose");

let UserSchema = new mongoose.Schema({
	Sequence:String,
	percent:String,
	exist:String,
	createTime:Date,
	lastLogin:Date,
})

mongoose.model('User',UserSchema);