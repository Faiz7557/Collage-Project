@extends('layouts.main')

@section('content')
    <h2 class="text-3xl font-bold text-white mb-6">Edit Data Pasar</h2>

    <div class="bg-white/30 backdrop-blur-xl p-8 rounded-2xl shadow-lg max-w-3xl mx-auto">
        <form action="{{ route('pasar.update', $pasar->id_pasar) }}" method="POST" class="space-y-4">
            @csrf
            @method('PUT')
            <div>
                <label class="block text-white mb-1 font-semibold">Nama Pasar</label>
                <input type="text" name="nama_pasar" value="{{ $pasar->nama_pasar }}" class="w-full p-2 rounded bg-white/20 border border-white/30 text-white placeholder-gray-300 focus:outline-none focus:ring-2 focus:ring-green-400">
            </div>
            <div>
                <label class="block text-white mb-1 font-semibold">User Penanggung Jawab</label>
                <select name="id_user" class="w-full p-2 rounded bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-green-400">
                    <option class="text-black">Pilih User</option>
                    @foreach($users as $user)
                        <option value="{{ $user->id_user }}" {{ $pasar->id_user == $user->id_user ? 'selected' : '' }} class="text-black">{{ $user->nama }}</option>
                    @endforeach
                </select>
            </div>
            <div class="flex justify-end space-x-4 pt-4">
                <a href="{{ route('pasar.index') }}" class="bg-gray-500 text-white px-5 py-2 rounded-lg font-bold hover:bg-gray-600 transition duration-300">Batal</a>
                <button type="submit" class="bg-green-600 text-white px-5 py-2 rounded-lg font-bold hover:bg-green-700 transition duration-300">Update</button>
            </div>
        </form>
    </div>
@endsection