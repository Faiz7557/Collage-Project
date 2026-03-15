<?php
namespace App\Http\Controllers;
use App\Models\Petani;
use App\Models\User;
use Illuminate\Http\Request;
class PetaniController extends Controller {
    public function index() {
        $petani = Petani::latest()->get();
        return view('petani.index', compact('petani'));
    }
    public function create() {
        $users = User::all();
        return view('petani.create', compact('users'));
    }
    public function store(Request $request) {
        $request->validate(['nama'=>'required','lokasi'=>'required','kontak'=>'required','kapasitas_panen'=>'required|integer','id_user'=>'required|exists:users,id_user']);
        Petani::create($request->all());
        return redirect()->route('petani.index')->with('success', 'Data Petani berhasil ditambahkan.');
    }
    public function show(Petani $petani){}
    public function edit(Petani $petani) {
        $users = User::all();
        return view('petani.edit', compact('petani', 'users'));
    }
    public function update(Request $request, Petani $petani) {
        $request->validate(['nama'=>'required','lokasi'=>'required','kontak'=>'required','kapasitas_panen'=>'required|integer','id_user'=>'required|exists:users,id_user']);
        $petani->update($request->all());
        return redirect()->route('petani.index')->with('success', 'Data Petani berhasil diperbarui.');
    }
    public function destroy(Petani $petani) {
        $petani->delete();
        return redirect()->route('petani.index')->with('success', 'Data Petani berhasil dihapus.');
    }
}