<?php
namespace App\Http\Controllers;
use App\Models\Pengepul;
use App\Models\User;
use Illuminate\Http\Request;
class PengepulController extends Controller {
    public function index(){ $pengepul = Pengepul::latest()->get(); return view('pengepul.index', compact('pengepul')); }
    public function create(){ $users = User::all(); return view('pengepul.create', compact('users')); }
    public function store(Request $request){
        $request->validate(['nama'=>'required','lokasi'=>'required','kapasitas_tampung'=>'required|integer','id_user'=>'required|exists:users,id_user']);
        Pengepul::create($request->all());
        return redirect()->route('pengepul.index')->with('success', 'Data Pengepul berhasil ditambahkan.');
    }
    public function show(Pengepul $pengepul){}
    public function edit(Pengepul $pengepul){ $users = User::all(); return view('pengepul.edit', compact('pengepul', 'users')); }
    public function update(Request $request, Pengepul $pengepul){
        $request->validate(['nama'=>'required','lokasi'=>'required','kapasitas_tampung'=>'required|integer','id_user'=>'required|exists:users,id_user']);
        $pengepul->update($request->all());
        return redirect()->route('pengepul.index')->with('success', 'Data Pengepul berhasil diperbarui.');
    }
    public function destroy(Pengepul $pengepul){ $pengepul->delete(); return redirect()->route('pengepul.index')->with('success', 'Data Pengepul berhasil dihapus.'); }
}