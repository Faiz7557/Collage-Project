<?php
namespace App\Http\Controllers;
use App\Models\Distributor;
use App\Models\User;
use Illuminate\Http\Request;
class DistributorController extends Controller {
    public function index(){ $distributor = Distributor::latest()->get(); return view('distributor.index', compact('distributor')); }
    public function create(){ $users = User::all(); return view('distributor.create', compact('users')); }
    public function store(Request $request){
        $request->validate(['nama'=>'required','wilayah_distribusi'=>'required','id_user'=>'required|exists:users,id_user']);
        Distributor::create($request->all());
        return redirect()->route('distributor.index')->with('success', 'Data Distributor berhasil ditambahkan.');
    }
    public function show(Distributor $distributor){}
    public function edit(Distributor $distributor){ $users = User::all(); return view('distributor.edit', compact('distributor', 'users')); }
    public function update(Request $request, Distributor $distributor){
        $request->validate(['nama'=>'required','wilayah_distribusi'=>'required','id_user'=>'required|exists:users,id_user']);
        $distributor->update($request->all());
        return redirect()->route('distributor.index')->with('success', 'Data Distributor berhasil diperbarui.');
    }
    public function destroy(Distributor $distributor){ $distributor->delete(); return redirect()->route('distributor.index')->with('success', 'Data Distributor berhasil dihapus.'); }
}