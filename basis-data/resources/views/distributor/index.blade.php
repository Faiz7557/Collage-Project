@extends('layouts.main')
@section('content')
    <div class="flex justify-between items-center mb-6"><h2 class="text-3xl font-bold text-white">Manajemen Data Distributor</h2><a href="{{ route('distributor.create') }}" class="bg-green-600 text-white px-5 py-2 rounded-lg font-bold hover:bg-green-700">Tambah Distributor</a></div>
    @if ($message = Session::get('success')) <div class="bg-green-200 text-green-800 p-4 rounded-lg mb-4">{{ $message }}</div> @endif
    <div class="bg-white/30 backdrop-blur-xl p-6 rounded-2xl">
        <table class="w-full text-left text-white">
            <thead><tr class="border-b border-white/20"><th class="p-4">Nama</th><th class="p-4">Wilayah Distribusi</th><th class="p-4">Aksi</th></tr></thead>
            <tbody>
                @forelse($distributor as $d)
                <tr class="border-b border-white/10">
                    <td class="p-4">{{ $d->nama }}</td><td class="p-4">{{ $d->wilayah_distribusi }}</td>
                    <td class="p-4"><form action="{{ route('distributor.destroy', $d->id) }}" method="POST" class="flex space-x-2">
                        <a href="{{ route('distributor.edit', $d->id) }}" class="bg-yellow-500 text-white px-3 py-1 rounded-lg font-semibold hover:bg-yellow-600">Edit</a>
                        @csrf @method('DELETE')
                        <button type="submit" class="bg-red-600 text-white px-3 py-1 rounded-lg font-semibold hover:bg-red-700" onclick="return confirm('Yakin?')">Hapus</button>
                    </form></td>
                </tr>
                @empty
                <tr><td colspan="3" class="text-center p-4">Belum ada data.</td></tr>
                @endforelse
            </tbody>
        </table>
    </div>
@endsection