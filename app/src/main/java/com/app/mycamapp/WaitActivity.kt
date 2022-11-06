package com.app.mycamapp

import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.ImageView
import com.bumptech.glide.Glide
import com.bumptech.glide.load.engine.DiskCacheStrategy

class WaitActivity : AppCompatActivity() {
    lateinit var blood:ImageView
    lateinit var ecgimg:ImageView
   lateinit var handler:Handler
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_wait)
        blood=findViewById(R.id.blood)
        ecgimg=findViewById(R.id.gifImage)

        val anim1=ObjectAnimator.ofPropertyValuesHolder(blood, PropertyValuesHolder.ofFloat("scaleX",1.15f),
            PropertyValuesHolder.ofFloat("scaleY",1.2f),PropertyValuesHolder.ofFloat("alpha",1f))
        anim1.duration=650
        anim1.repeatCount=ValueAnimator.INFINITE
        anim1.repeatMode=ValueAnimator.REVERSE
        anim1.start()
        Glide.with(this).asGif().load(R.drawable.ecg3)
            .diskCacheStrategy(DiskCacheStrategy.ALL)
            .into(ecgimg)
        handler= Handler()
        handler.postDelayed({
            val intent = Intent(this,upload::class.java)
            startActivity(intent)
            finish()
        },15000)



    }



}