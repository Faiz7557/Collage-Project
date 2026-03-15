@extends('layouts.app')

@section('content')
<div class="container mx-auto px-4 py-8">
    <div class="mb-6">
        <a href="{{ route('pasar.index') }}" class="text-green-500 hover:text-green-600">
            &larr; Kembali ke Pasar
        </a>
    </div>

    <div class="bg-white rounded-lg shadow-lg p-6">
        <h1 class="text-2xl font-bold mb-6">Tambah Produk Beras</h1>
        
        <form action="{{ route('pasar.store') }}" method="POST" enctype="multipart/form-data">
            @csrf
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                    <label for="nama_produk" class="block text-sm font-medium text-gray-700 mb-1">Nama Produk</label>
                    <input type="text" name="nama_produk" id="nama_produk" value="{{ old('nama_produk') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required>
                    @error('nama_produk')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div>
                    <label for="varietas" class="block text-sm font-medium text-gray-700 mb-1">Varietas</label>
                    <input type="text" name="varietas" id="varietas" value="{{ old('varietas') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required>
                    @error('varietas')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div>
                    <label for="kualitas" class="block text-sm font-medium text-gray-700 mb-1">Kualitas</label>
                    <input type="text" name="kualitas" id="kualitas" value="{{ old('kualitas') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required>
                    @error('kualitas')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div>
                    <label for="harga_per_kg" class="block text-sm font-medium text-gray-700 mb-1">Harga per Kg (Rp)</label>
                    <input type="number" name="harga_per_kg" id="harga_per_kg" value="{{ old('harga_per_kg') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required min="0">
                    @error('harga_per_kg')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div>
                    <label for="lokasi_gudang" class="block text-sm font-medium text-gray-700 mb-1">Lokasi Gudang</label>
                    <input type="text" name="lokasi_gudang" id="lokasi_gudang" value="{{ old('lokasi_gudang') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required>
                    @error('lokasi_gudang')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div>
                    <label for="stok_kg" class="block text-sm font-medium text-gray-700 mb-1">Stok (Kg)</label>
                    <input type="number" name="stok_kg" id="stok_kg" value="{{ old('stok_kg') }}" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        required min="1">
                    @error('stok_kg')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div class="md:col-span-2">
                    <label for="deskripsi" class="block text-sm font-medium text-gray-700 mb-1">Deskripsi</label>
                    <textarea name="deskripsi" id="deskripsi" rows="4" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200">{{ old('deskripsi') }}</textarea>
                    @error('deskripsi')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
                
                <div class="md:col-span-2">
                    <label for="foto" class="block text-sm font-medium text-gray-700 mb-1">Foto Produk</label>
                    <input type="file" name="foto" id="foto" 
                        class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200"
                        accept="image/jpeg,image/png,image/jpg">
                    <p class="text-sm text-gray-500 mt-1">Format: JPG, JPEG, PNG. Maks: 2MB</p>
                    @error('foto')
                        <p class="text-red-500 text-sm mt-1">{{ $message }}</p>
                    @enderror
                </div>
            </div>
            
            <div class="mt-6">
                <button type="submit" class="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-medium">
                    Simpan Produk
                </button>
            </div>
        </form>
    </div>
</div>
@endsection