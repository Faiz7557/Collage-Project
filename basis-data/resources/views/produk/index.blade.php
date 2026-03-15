@extends('layouts.app')

@section('content')
<div class="container mx-auto px-4 py-8">
    <div class="flex justify-between items-center mb-6">
        <h1 class="text-2xl font-bold text-gray-800">Daftar Produk Beras</h1>
        <a href="{{ route('produk.create') }}" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg shadow-md transition duration-300 ease-in-out">
            Tambah Produk
        </a>
    </div>

    @if(session('success'))
    <div class="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded-lg mb-6" role="alert">
        {{ session('success') }}
    </div>
    @endif

    <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        @foreach($produk as $item)
        <div class="bg-white rounded-xl shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300 ease-in-out">
            @if($item->foto)
            <img src="{{ asset('storage/produk/' . $item->foto) }}" alt="{{ $item->nama_produk }}" class="w-full h-48 object-cover">
            @else
            <div class="w-full h-48 bg-gray-200 flex items-center justify-center text-gray-500 text-sm">
                Tidak ada foto
            </div>
            @endif
            <div class="p-5">
                <h2 class="text-xl font-semibold text-gray-800 mb-2">{{ $item->nama_produk }}</h2>
                <p class="text-gray-600 text-sm mb-1">Jenis: {{ $item->jenis_beras }}</p>
                <p class="text-gray-600 text-sm mb-1">Petani: {{ $item->nama_petani }}</p>
                <p class="text-gray-600 text-sm mb-1">Lokasi: {{ $item->lokasi_gudang }}</p>
                <p class="text-gray-600 text-sm mb-3">Stok: {{ $item->stok }} kg</p>
                <p class="text-green-700 font-bold text-lg mb-4">Rp {{ number_format($item->harga, 0, ',', '.') }}</p>
                
                <div class="flex flex-col space-y-2">
                    <a href="{{ route('produk.show', $item->id_produk) }}" class="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg text-center text-sm font-medium transition duration-300 ease-in-out">Lihat Detail</a>
                    <div class="flex justify-between space-x-2">
                        <a href="{{ route('produk.edit', $item->id_produk) }}" class="flex-1 bg-yellow-500 hover:bg-yellow-600 text-white py-2 px-4 rounded-lg text-center text-sm font-medium transition duration-300 ease-in-out">Edit</a>
                        <form action="{{ route('produk.destroy', $item->id_produk) }}" method="POST" onsubmit="return confirm('Apakah Anda yakin ingin menghapus produk ini?')">
                            @csrf
                            @method('DELETE')
                            <button type="submit" class="flex-1 bg-red-500 hover:bg-red-600 text-white py-2 px-4 rounded-lg text-center text-sm font-medium transition duration-300 ease-in-out">Hapus</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
        @endforeach
    </div>
</div>
@endsection