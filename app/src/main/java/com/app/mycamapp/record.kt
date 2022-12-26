package com.app.mycamapp

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.ContactsContract.Data
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.widget.Button
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.*

class record : AppCompatActivity() {
    private lateinit var recordRecycler:RecyclerView
    private lateinit var  recordArray: ArrayList<String>
    private lateinit var database : DatabaseReference
    private lateinit var user: FirebaseAuth

    lateinit var madapter: recycleradapter
    var rec = ArrayList<user_rec>()
    lateinit var email:String


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_record)
       // email = intent.getStringExtra("Email").toString()
        var obj = DataUpdate()
        email=obj.getemail()
        database = FirebaseDatabase.getInstance().getReference("user_record")
        recordRecycler = findViewById(R.id.recyclerView)
        recordRecycler.layoutManager = LinearLayoutManager(this)
        getdata(email)





    }

    fun recordShow(){
        val intent = Intent(this,DataUpdate::class.java)
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

    private fun getdata(e:String)
    {   var i:Int = 0
        database.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {

                for(ds in snapshot.children)
                {
                    val id=ds.key

                    val email=ds.child("Email").value.toString()
                    val gl=ds.child("GLucose").value.toString()
                    val hb=ds.child("Hemo").value.toString()
                    val dt=ds.child("Date").value.toString()
                    val a = user_rec()
                    a.gl=gl
                    a.hb=hb
                    a.date=dt
                    if(email==e)
                    {
                        rec.add(i,a)
                        var y:String=rec[i].gl
                        Log.d("dfgdf",y)
                        i=i+1
                    }
                }
                rec.reverse()
                //Log.d("dfgdf","my name is sadman")
                madapter = recycleradapter(applicationContext,rec)
                recordRecycler.adapter = madapter
            }

            override fun onCancelled(error: DatabaseError) {

            }
        })

    }
}