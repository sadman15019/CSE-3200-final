package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener

class result : AppCompatActivity() {
    lateinit var gl_result:TextView
    lateinit var hb_result:TextView
    lateinit var btn:Button
    //lateinit var ref:DatabaseReference
    lateinit var yo:TextView
    lateinit var gl:String
    lateinit var hb:String
    lateinit var c:String
    lateinit var email:String
    private lateinit var user:FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_result)
        email = intent.getStringExtra("Email").toString()
        gl=intent.getStringExtra("gl").toString()
        hb=intent.getStringExtra("hb").toString()
        gl_result=findViewById(R.id.GL)
        hb_result=findViewById(R.id.Hb)
        btn = findViewById(R.id.savebtn)

        yo=findViewById(R.id.hbtext)
        /*val delim = ":"
        val arr = c.split(delim).toTypedArray()
        gl=arr[1].toString()
        hb=arr[0].toString()
        gl=gl.substring(2,5)
        hb=hb.substring(2,6)*/
        gl_result.text=gl
        hb_result.text=hb
        yo.text="Hb"
 btn.setOnClickListener {
     var intent = Intent(this, record::class.java)
     intent.putExtra("Email", email)
     startActivity(intent)

 }
    }

    fun recordShow(){
        val intent = Intent(this,record::class.java)
        startActivity(intent)
        finish()
    }
    fun fragmentShow(){
        val dialog = BottomSheetDialog(this)

        // on below line we are inflating a layout file which we have created.
        val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog, null)

        // on below line we are creating a variable for our button
        // which we are using to dismiss our dialog.
        val btnnext = view.findViewById<Button>(R.id.nextbtn)

        // on below line we are adding on click listener
        // for our dismissing the dialog button.
        btnnext.setOnClickListener {
            // on below line we are calling a dismiss
            // method to close our dialog.
            dialog.dismiss()
            val dialog = BottomSheetDialog(this)

            // on below line we are inflating a layout file which we have created.
            val view = layoutInflater.inflate(R.layout.bottom_sheet_dialog2, null)

            // on below line we are creating a variable for our button
            // which we are using to dismiss our dialog.
            val btndone = view.findViewById<Button>(R.id.donebtn)

            // on below line we are adding on click listener
            // for our dismissing the dialog button.
            btndone.setOnClickListener {
                // on below line we are calling a dismiss
                // method to close our dialog.
                dialog.dismiss()
            }
            // below line is use to set cancelable to avoid
            // closing of dialog box when clicking on the screen.
            dialog.setCancelable(false)

            // on below line we are setting
            // content view to our view.
            dialog.setContentView(view)

            // on below line we are calling
            // a show method to display a dialog.
            dialog.show()
        }
        // below line is use to set cancelable to avoid
        // closing of dialog box when clicking on the screen.
        dialog.setCancelable(false)

        // on below line we are setting
        // content view to our view.
        dialog.setContentView(view)

        // on below line we are calling
        // a show method to display a dialog.
        dialog.show()
    }
    fun signOut(){
        user= FirebaseAuth.getInstance()
        user.signOut()
        val intent = Intent(this,StartupActivity::class.java)
        startActivity(intent)
        finish()
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu,menu);
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        val id = item.itemId;

        if (id == R.id.logout_action){

            signOut()
            return true
        }
        if(id==R.id.settings_action){
            // on below line we are creating a new bottom sheet dialog.


            fragmentShow()
            return true
        }
        if(id==R.id.record_action){

            recordShow()
            return true

        }

        return super.onOptionsItemSelected(item)
    }

}