@extends('layouts.main')

@section('content')
<div class="container-fluid p-4">
    <!-- Header Section -->
    <div class="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h1 class="h3 font-weight-bold text-dark mb-1" style="font-family: 'Playfair Display', serif;">Manage Products</h1>
            <p class="text-secondary mb-0">Kelola katalog produk beras di Warung Padi.</p>
        </div>
        <a href="{{ route('admin.products.create') }}" class="btn btn-rice-green rounded-pill px-4 py-2 shadow-sm fw-bold">
            <i class="bi bi-plus-lg me-2"></i> Tambah Produk
        </a>
    </div>

    @if(session('success'))
        <div class="alert alert-success alert-dismissible fade show shadow-sm border-0 rounded-4 mb-4" role="alert" style="background-color: #d1e7dd; color: #0f5132;">
            <i class="bi bi-check-circle-fill me-2"></i> {{ session('success') }}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    @endif

    <!-- Main Card -->
    <div class="card border-0 shadow-lg rounded-4 overflow-hidden" style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px);">
        <!-- Card Header with Theme Gradient -->
        <div class="card-header border-0 p-4 d-flex justify-content-between align-items-center" 
             style="background: linear-gradient(135deg, #198754, #81C784);">
            <div class="d-flex align-items-center text-white">
                <i class="bi bi-box-seam-fill fs-4 me-3 opacity-75"></i>
                <h5 class="fw-bold mb-0" style="letter-spacing: 0.5px;">Daftar Produk</h5>
            </div>
            
            <div class="d-none d-md-block" style="width: 300px;">
                <div class="input-group">
                    <span class="input-group-text border-0 bg-white text-muted ps-3" style="border-radius: 20px 0 0 20px;">
                        <i class="bi bi-search"></i>
                    </span>
                    <input type="text" class="form-control border-0 bg-white text-dark" placeholder="Cari produk..." style="border-radius: 0 20px 20px 0; font-size: 0.9rem;">
                </div>
            </div>
        </div>

        <div class="card-body p-0">
            <div class="table-responsive">
                <table class="table table-hover align-middle mb-0">
                    <thead style="background-color: #f8f9fa;">
                        <tr>
                            <th class="ps-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Produk</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Petani</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Harga / Kg</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Stok</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Lokasi</th>
                            <th class="text-end pe-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Aksi</th>
                        </tr>
                    </thead>
                    <tbody>
                        @forelse($products as $product)
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05); transition: background-color 0.2s;">
                            <td class="ps-4 py-3">
                                <div class="d-flex align-items-center">
                                    <div class="me-3 position-relative">
                                        @if($product->foto)
                                            <img src="{{ asset('storage/' . $product->foto) }}" alt="{{ $product->nama_produk }}" class="rounded-3 shadow-sm" style="width: 60px; height: 60px; object-fit: cover;">
                                        @else
                                            <div class="bg-light rounded-3 d-flex align-items-center justify-content-center text-secondary border" style="width: 60px; height: 60px;">
                                                <i class="bi bi-image fs-4 opacity-50"></i>
                                            </div>
                                        @endif
                                    </div>
                                    <div>
                                        <div class="fw-bold text-dark fs-6">{{ $product->nama_produk }}</div>
                                        <div class="small text-muted">{{ $product->jenis_beras }} • {{ $product->kualitas }}</div>
                                    </div>
                                </div>
                            </td>
                            <td>
                                <div class="d-flex align-items-center">
                                    <div class="avatar-sm rounded-circle bg-success-subtle text-success d-flex align-items-center justify-content-center me-2" style="width: 32px; height: 32px; font-size: 0.8rem; font-weight: bold;">
                                        {{ substr($product->petani->nama ?? $product->nama_petani ?? 'U', 0, 1) }}
                                    </div>
                                    <span class="text-dark fw-medium small">{{ $product->petani->nama ?? $product->nama_petani ?? 'Unknown' }}</span>
                                </div>
                            </td>
                            <td>
                                <span class="fw-bold text-success">Rp {{ number_format($product->harga, 0, ',', '.') }}</span>
                            </td>
                            <td>
                                <span class="badge {{ $product->stok > 100 ? 'bg-info-subtle text-info' : 'bg-warning-subtle text-warning' }} rounded-pill px-3">
                                    {{ number_format($product->stok, 0, ',', '.') }} Kg
                                </span>
                            </td>
                            <td class="text-secondary small">
                                <i class="bi bi-geo-alt me-1 text-danger"></i> {{ Str::limit($product->lokasi_gudang, 20) }}
                            </td>
                            <td class="text-end pe-4">
                                <div class="btn-group shadow-sm rounded-3 overflow-hidden">
                                    <a href="{{ route('admin.products.edit', $product->id_produk) }}" class="btn btn-sm btn-action bg-white text-primary border-0" data-bs-toggle="tooltip" title="Edit">
                                        <i class="bi bi-pencil-square"></i>
                                    </a>
                                    <form action="{{ route('admin.products.destroy', $product->id_produk) }}" method="POST" class="d-inline" onsubmit="return confirm('Apakah Anda yakin ingin menghapus produk ini?');">
                                        @csrf
                                        @method('DELETE')
                                        <button type="submit" class="btn btn-sm btn-action bg-white text-danger border-0" data-bs-toggle="tooltip" title="Delete">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                        @empty
                        <tr>
                            <td colspan="6" class="text-center py-5">
                                <div class="d-flex flex-column align-items-center justify-content-center">
                                    <img src="https://cdni.iconscout.com/illustration/premium/thumb/product-not-found-3428101-2859740.png" alt="Empty" style="width: 150px; opacity: 0.6; mix-blend-mode: multiply;">
                                    <h5 class="text-muted mt-3">Belum ada produk terdaftar</h5>
                                    <p class="text-secondary small">Silakan tambahkan produk baru untuk memulai.</p>
                                </div>
                            </td>
                        </tr>
                        @endforelse
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card-footer bg-white border-0 py-4 px-4">
            {{ $products->links() }}
        </div>
    </div>
</div>

<style>
    /* Custom Styling for refined theme */
    :root {
        --rice-green: #4CAF50;
        --rice-gold: #FF9800;
        --font-heading: 'Playfair Display', serif;
    }

    /* FORCE OVERRIDE GLOBAL TEXT COLORS FOR THIS PAGE */
    .card.bg-white, .card[style*="background: rgba(255, 255, 255"] {
        color: #212529 !important;
    }

    .card .text-dark {
        color: #212529 !important;
    }
    
    .card .text-muted {
        color: #6c757d !important;
    }

    .table {
        color: #212529 !important;
    }
    
    .table thead th {
        color: #6c757d !important;
        border-bottom: 2px solid #eee !important;
    }
    
    .table td {
        border-bottom: 1px solid #f2f2f2 !important;
        color: #212529 !important; /* Ensure content is dark */
    }
    
    .fw-bold.text-dark {
         color: #212529 !important;
    }

    /* end overrides */

    .btn-rice-green {
        background: linear-gradient(135deg, #4CAF50, #81C784);
        color: white;
        border: none;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .btn-rice-green:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3) !important;
        color: white;
    }

    .btn-action {
        transition: background-color 0.2s;
    }
    .btn-action:hover {
        background-color: #f8f9fa !important;
    }
    
    tr:hover {
        background-color: rgba(255, 252, 230, 0.4) !important; /* Subtle warm hover */
    }

    /* Pagination Customization */
    .pagination {
        margin-bottom: 0;
        gap: 5px;
        justify-content: flex-end;
    }
    .pagination .page-item .page-link {
        border-radius: 8px;
        border: 1px solid #eee;
        color: #555;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .pagination .page-item.active .page-link {
        background: var(--rice-green);
        border-color: var(--rice-green);
        color: white;
        box-shadow: 0 4px 10px rgba(76, 175, 80, 0.3);
    }
    .pagination .page-item.disabled .page-link {
        background-color: #f9f9f9;
        border-color: #eee;
    }
</style>
@endsection
