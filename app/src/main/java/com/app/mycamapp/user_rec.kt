package com.app.mycamapp

class user_rec {
    var gl:String =""
    var hb:String =""
    var date:String=""
   fun setgl(gl:String)
    {
        this.gl=gl
    }
    fun sethb(hb:String) {
     this.hb=hb
    }
    fun getgl():String
    {
        return gl
    }
    fun gethb():String
    {
        return hb
    }
}