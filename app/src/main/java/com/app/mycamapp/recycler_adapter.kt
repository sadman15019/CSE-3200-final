package com.app.mycamapp

import android.content.Context
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.recyclerview.widget.RecyclerView
import com.example.mycamapp.User
import org.w3c.dom.Text


class recycleradapter(val context: Context,val items:ArrayList<user_rec>): RecyclerView.Adapter<recycleradapter.viewHolder>()
{

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): viewHolder {
        return viewHolder( LayoutInflater.from(context).inflate(R.layout.show_record, parent, false))
    }

    override fun onBindViewHolder(holder: viewHolder, position: Int) {
       val item:user_rec=items[position]
        Log.d("fg","item size"+item.gl)
        //holder.mdate.text=item
        holder.mgl.text=item.gl
        holder.mhb.text=item.hb
        holder.mdate.text=item.date
    }

    override fun getItemCount(): Int {
       return  items.size
    }
    class viewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {



        //var mdate:TextView = itemView.findViewById(R.id.date)
        var mgl:TextView = itemView.findViewById(R.id.sys_m)
        var mhb:TextView = itemView.findViewById(R.id.dia_m)
        var mdate:TextView = itemView.findViewById(R.id.date)
        //var mLayout:TextView = itemView.findViewById(R.id.loglayout)
    }
}
