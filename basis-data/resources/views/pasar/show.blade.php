@extends('layouts.app')

@section('content')
<div class="container mx-auto px-4 py-8">
    <div class="mb-6">
        <a href="{{ route('pasar.index') }}" class="text-green-500 hover:text-green-600">
            &larr; Kembali ke Pasar
        </a>
    </div>

    <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="md:flex">
            <div class="md:w-1/2">
                <div class="h-96 bg-gray-200">
                    @if($produk->foto)
                        <img src="{{ Storage::url($produk->foto) }}" alt="{{ $produk->nama_produk }}" class="w-full h-full object-cover">
                    @else
                        <div class="w-full h-full flex items-center justify-center bg-gray-100">
                            <span class="text-gray-400">Tidak ada foto</span>
                        </div>
                    @endif
                </div>
            </div>
            
            <div class="md:w-1/2 p-6">
                <h1 class="text-3xl font-bold mb-4">{{ $produk->nama_produk }}</h1>
                
                <div class="mb-4">
                    <div class="flex items-center">
                        <span class="text-yellow-400">
                            @for($i = 1; $i <= 5; $i++)
                                @if($i <= round($avgRating))
                                    ★
                                @else
                                    ☆
                                @endif
                            @endfor
                        </span>
                        <span class="ml-2 text-gray-600">{{ number_format($avgRating, 1) }} ({{ $produk->rating->count() }} ulasan)</span>
                    </div>
                </div>
                
                <div class="mb-4">
                    <p class="text-3xl font-bold text-green-600">Rp {{ number_format($produk->harga_per_kg, 0, ',', '.') }}/kg</p>
                </div>
                
                <div class="grid grid-cols-2 gap-4 mb-6">
                    <div>
                        <h3 class="text-sm text-gray-600">Varietas</h3>
                        <p class="font-medium">{{ $produk->varietas }}</p>
                    </div>
                    
                    <div>
                        <h3 class="text-sm text-gray-600">Kualitas</h3>
                        <p class="font-medium">{{ $produk->kualitas }}</p>
                    </div>
                    
                    <div>
                        <h3 class="text-sm text-gray-600">Stok Tersedia</h3>
                        <p class="font-medium">{{ number_format($produk->stok_kg, 0, ',', '.') }} kg</p>
                    </div>
                    
                    <div>
                        <h3 class="text-sm text-gray-600">Lokasi Gudang</h3>
                        <p class="font-medium">{{ $produk->lokasi_gudang }}</p>
                    </div>
                    
                    <div>
                        <h3 class="text-sm text-gray-600">Petani</h3>
                        <p class="font-medium">{{ $produk->petani->nama }}</p>
                    </div>
                </div>
                
                <div class="mb-6">
                    <h3 class="text-lg font-semibold mb-2">Deskripsi</h3>
                    <p class="text-gray-700">{{ $produk->deskripsi ?? 'Tidak ada deskripsi' }}</p>
                </div>
                
                <div class="flex flex-wrap gap-2">
                    @if(auth()->user()->peran === 'pengepul')
                        <a href="{{ route('negosiasi.create', $produk) }}" class="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-medium">
                            Ajukan Negosiasi
                        </a>
                    @endif
                    
                    @if(auth()->id() === $produk->id_petani)
                        <a href="{{ route('pasar.edit', $produk) }}" class="bg-blue-500 hover:bg-blue-600 text-white px-6 py-3 rounded-lg font-medium">
                            Edit Produk
                        </a>
                        
                        <form action="{{ route('pasar.destroy', $produk) }}" method="POST" class="inline" onsubmit="return confirm('Apakah Anda yakin ingin menghapus produk ini?')">
                            @csrf
                            @method('DELETE')
                            <button type="submit" class="bg-red-500 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-medium">
                                Hapus Produk
                            </button>
                        </form>
                    @endif
                </div>
            </div>
        </div>
        
        <!-- Rating dan Ulasan -->
        <div class="p-6 border-t">
            <h2 class="text-xl font-semibold mb-4">Rating dan Ulasan</h2>
            
            @if(auth()->user()->peran !== 'petani' || auth()->id() !== $produk->id_petani)
                <div class="mb-6 bg-gray-50 p-4 rounded-lg">
                    <h3 class="text-lg font-medium mb-2">Berikan Rating</h3>
                    <form action="{{ route('pasar.rate', $produk) }}" method="POST">
                        @csrf
                        <div class="mb-4">
                            <label for="nilai_rating" class="block text-sm font-medium text-gray-700 mb-1">Rating</label>
                            <select name="nilai_rating" id="nilai_rating" class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200">
                                <option value="5">5 - Sangat Baik</option>
                                <option value="4">4 - Baik</option>
                                <option value="3">3 - Cukup</option>
                                <option value="2">2 - Kurang</option>
                                <option value="1">1 - Sangat Kurang</option>
                            </select>
                        </div>
                        
                        <div class="mb-4">
                            <label for="komentar" class="block text-sm font-medium text-gray-700 mb-1">Komentar</label>
                            <textarea name="komentar" id="komentar" rows="3" class="w-full border-gray-300 rounded-md shadow-sm focus:border-green-500 focus:ring focus:ring-green-200" placeholder="Tulis komentar Anda..."></textarea>
                        </div>
                        
                        <button type="submit" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded">
                            Kirim Rating
                        </button>
                    </form>
                </div>
            @endif
            
            <div>
                @forelse($produk->rating as $rating)
                    <div class="mb-4 pb-4 border-b">
                        <div class="flex justify-between items-start">
                            <div>
                                <div class="flex items-center">
                                    <span class="font-medium">{{ $rating->user->nama }}</span>
                                    <span class="mx-2 text-gray-400">•</span>
                                    <span class="text-yellow-400">
                                        @for($i = 1; $i <= 5; $i++)
                                            @if($i <= $rating->nilai_rating)
                                                ★
                                            @else
                                                ☆
                                            @endif
                                        @endfor
                                    </span>
                                </div>
                                <p class="text-sm text-gray-500">{{ $rating->created_at->format('d M Y, H:i') }}</p>
                            </div>
                        </div>
                        
                        @if($rating->komentar)
                            <p class="mt-2 text-gray-700">{{ $rating->komentar }}</p>
                        @endif
                    </div>
                @empty
                    <p class="text-gray-500">Belum ada ulasan untuk produk ini.</p>
                @endforelse
            </div>
        </div>
    </div>
</div>
@endsection