@extends('layouts.main')

@section('content')
    <div class="flex justify-between items-center mb-6">
        <h2 class="text-3xl font-bold text-white">Manajemen Data Petani</h2>
        <a href="{{ route('petani.create') }}" class="bg-green-600 text-white px-5 py-2 rounded-lg font-bold hover:bg-green-700 transition duration-300">
            Tambah Petani
        </a>
    </div>

    @if ($message = Session::get('success'))
        <div class="bg-green-500 text-white p-4 rounded-lg mb-6 shadow-lg">
            {{ $message }}
        </div>
    @endif

    <div class="bg-white/30 backdrop-blur-xl p-6 rounded-2xl shadow-lg">
        <table class="w-full text-left text-white">
            <thead>
                <tr class="border-b border-white/20">
                    <th class="p-4 font-bold">Nama</th>
                    <th class="p-4 font-bold">Lokasi</th>
                    <th class="p-4 font-bold">Kontak</th>
                    <th class="p-4 font-bold">Kapasitas (Kg)</th>
                    <th class="p-4 font-bold">Aksi</th>
                </tr>
            </thead>
            <tbody>
                @forelse($petani as $p)
                <tr class="border-b border-white/10 hover:bg-white/10">
                    <td class="p-4">{{ $p->nama }}</td>
                    <td class="p-4">{{ $p->lokasi }}</td>
                    <td class="p-4">{{ $p->kontak }}</td>
                    <td class="p-4">{{ number_format($p->kapasitas_panen) }}</td>
                    <td class="p-4">
                        <form action="{{ route('petani.destroy', $p->id) }}" method="POST" class="flex space-x-2">
                            <a href="{{ route('petani.edit', $p->id) }}" class="bg-yellow-500 text-white px-3 py-1 rounded-lg font-semibold hover:bg-yellow-600 transition duration-300">Edit</a>
                            @csrf
                            @method('DELETE')
                            <button type="submit" class="bg-red-600 text-white px-3 py-1 rounded-lg font-semibold hover:bg-red-700 transition duration-300" onclick="return confirm('Apakah Anda yakin ingin menghapus data ini?')">Hapus</button>
                        </form>
                    </td>
                </tr>
                @empty
                <tr>
                    <td colspan="5" class="text-center p-4">Belum ada data.</td>
                </tr>
                @endforelse
            </tbody>
        </table>
    </div>
@endsection