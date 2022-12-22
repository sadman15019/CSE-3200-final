package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button
import android.widget.ImageButton
import com.google.android.material.bottomsheet.BottomSheetDialog

class instruction : AppCompatActivity() {
     lateinit var age:String
     lateinit var gender:String
    lateinit var nid:String
    lateinit var name:String
     lateinit var appcambtn:Button
    // creating a variable for our button
    lateinit var btnShowBottomSheet: ImageButton
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_instruction)
        nid=intent.getStringExtra("nid").toString()
        name=intent.getStringExtra("name").toString()
        age = intent.getStringExtra("age").toString()
        gender = intent.getStringExtra("gender").toString()
        appcambtn=findViewById(R.id.appcambtn)
        appcambtn.setOnClickListener {
            val intent = Intent(this,MainActivity::class.java)
            intent.putExtra("nid",nid)
            intent.putExtra("name",name)
            intent.putExtra("gender",gender)
            intent.putExtra("age",age)
            startActivity(intent)
            finish()
        }
        // initializing our variable for button with its id.
        btnShowBottomSheet = findViewById(R.id.idBtnShowBottomSheet);

        // adding on click listener for our button.
        btnShowBottomSheet.setOnClickListener {

            // on below line we are creating a new bottom sheet dialog.
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
                nextbottomsheet()
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
    }
    private fun nextbottomsheet() {  // on below line we are creating a new bottom sheet dialog.
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

}