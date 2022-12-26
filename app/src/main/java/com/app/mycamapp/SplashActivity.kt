package com.app.mycamapp

import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.animation.ValueAnimator
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.widget.ImageView


class SplashActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        lateinit var blood: ImageView

        lateinit var handler: Handler

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)

            blood=findViewById(R.id.blood2)


            val anim1= ObjectAnimator.ofPropertyValuesHolder(blood, PropertyValuesHolder.ofFloat("scaleX",1.15f),
                PropertyValuesHolder.ofFloat("scaleY",1.2f), PropertyValuesHolder.ofFloat("alpha",1f))
            anim1.duration=650
            anim1.repeatCount= ValueAnimator.INFINITE
            anim1.repeatMode= ValueAnimator.REVERSE
            anim1.start()
            handler= Handler()
            handler.postDelayed({
                val intent = Intent(this,StartupActivity::class.java)
                startActivity(intent)
                finish()
            },2500)


        }
}