package com.app.mycamapp

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.ContactsContract.Data
import android.util.Log
import android.view.Menu
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.firebase.database.*

class record : AppCompatActivity() {
    private lateinit var recordRecycler:RecyclerView
    private lateinit var  recordArray: ArrayList<String>
    private lateinit var database : DatabaseReference
    lateinit var madapter: recycleradapter
    var rec = ArrayList<user_rec>()
    lateinit var email:String


    override fun onCreate(savedInstanceState: Bundle?) {

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_record)
        email = intent.getStringExtra("Email").toString()
        database = FirebaseDatabase.getInstance().getReference("user_record")
        recordRecycler = findViewById(R.id.recyclerView)
        recordRecycler.layoutManager = LinearLayoutManager(this)
        getdata(email)





    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        menuInflater.inflate(R.menu.menu,menu);
        return true
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
                Log.d("dfgdf","my name is sadman")
                madapter = recycleradapter(applicationContext,rec)
                recordRecycler.adapter = madapter
            }

            override fun onCancelled(error: DatabaseError) {

            }
        })

    }
}