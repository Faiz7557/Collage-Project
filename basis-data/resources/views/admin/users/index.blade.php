@extends('layouts.main')

@section('content')
<div class="container-fluid p-4">
    <!-- Header Section -->
    <div class="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h1 class="h3 font-weight-bold text-dark mb-1" style="font-family: 'Playfair Display', serif;">Manage Users</h1>
            <p class="text-secondary mb-0">Kelola data pengguna ekosistem Warung Padi.</p>
        </div>
        <a href="{{ route('admin.users.create') }}" class="btn btn-rice-green rounded-pill px-4 py-2 shadow-sm fw-bold">
            <i class="bi bi-plus-lg me-2"></i> Tambah User Baru
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
                <i class="bi bi-people-fill fs-4 me-3 opacity-75"></i>
                <h5 class="fw-bold mb-0" style="letter-spacing: 0.5px;">Daftar Pengguna</h5>
            </div>
            
            <div class="d-none d-md-block" style="width: 300px;">
                <div class="input-group">
                    <span class="input-group-text border-0 bg-white text-muted ps-3" style="border-radius: 20px 0 0 20px;">
                        <i class="bi bi-search"></i>
                    </span>
                    <input type="text" class="form-control border-0 bg-white text-dark" placeholder="Cari nama atau email..." style="border-radius: 0 20px 20px 0; font-size: 0.9rem;">
                </div>
            </div>
        </div>

        <div class="card-body p-0">
            <div class="table-responsive">
                <table class="table table-hover align-middle mb-0">
                    <thead style="background-color: #f8f9fa;">
                        <tr>
                            <th class="ps-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">No</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Pengguna</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Email</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Role</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Terdaftar</th>
                            <th class="text-end pe-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Aksi</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach($users as $index => $user)
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05); transition: background-color 0.2s;">
                            <td class="ps-4 fw-bold text-muted">{{ $users->firstItem() + $index }}</td>
                            <td class="py-3">
                                <div class="d-flex align-items-center">
                                    <div class="avatar-sm rounded-circle shadow-sm d-flex align-items-center justify-content-center me-3" 
                                         style="width: 45px; height: 45px; background: linear-gradient(135deg, #FF9800, #FFCC80); color: white; font-weight: bold; font-size: 1.1rem;">
                                        {{ substr($user->nama, 0, 1) }}
                                    </div>
                                    <div>
                                        <div class="fw-bold text-dark" style="font-size: 0.95rem;">{{ $user->nama }}</div>
                                        <div class="small text-muted d-block d-md-none">{{ $user->email }}</div>
                                    </div>
                                </div>
                            </td>
                            <td class="text-dark">{{ $user->email }}</td>
                            <td>
                                @php
                                    $roleStyles = match($user->peran) {
                                        'admin' => 'background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2;',
                                        'petani' => 'background-color: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9;',
                                        'pengepul' => 'background-color: #fff3e0; color: #ef6c00; border: 1px solid #ffe0b2;',
                                        'distributor' => 'background-color: #e3f2fd; color: #1565c0; border: 1px solid #bbdefb;',
                                        default => 'background-color: #f5f5f5; color: #616161; border: 1px solid #e0e0e0;'
                                    };
                                @endphp
                                <span class="badge rounded-pill px-3 py-2 fw-semibold" style="{{ $roleStyles }}">
                                    {{ ucfirst($user->peran) }}
                                </span>
                            </td>
                            <td class="text-muted">
                                <span style="font-size: 0.9rem;">{{ $user->created_at->format('d M Y') }}</span>
                            </td>
                            <td class="text-end pe-4">
                                <div class="btn-group shadow-sm rounded-3 overflow-hidden">
                                    <a href="{{ route('admin.users.edit', $user) }}" class="btn btn-sm btn-action bg-white text-primary border-0" data-bs-toggle="tooltip" title="Edit">
                                        <i class="bi bi-pencil-square"></i>
                                    </a>
                                    <form action="{{ route('admin.users.destroy', $user) }}" method="POST" class="d-inline" onsubmit="return confirm('Apakah Anda yakin ingin menghapus pengguna ini?');">
                                        @csrf
                                        @method('DELETE')
                                        <button type="submit" class="btn btn-sm btn-action bg-white text-danger border-0" data-bs-toggle="tooltip" title="Delete">
                                            <i class="bi bi-trash"></i>
                                        </button>
                                    </form>
                                </div>
                            </td>
                        </tr>
                        @endforeach
                    </tbody>
                </table>
            </div>

            @if($users->isEmpty())
                <div class="text-center py-5">
                    <div class="mb-3">
                        <img src="https://cdni.iconscout.com/illustration/premium/thumb/search-result-not-found-2130361-1800925.png" alt="Empty" style="width: 120px; opacity: 0.7;">
                    </div>
                    <h5 class="text-muted">Belum ada pengguna ditemukan</h5>
                </div>
            @endif
        </div>
        
        <div class="card-footer bg-white border-0 py-4 px-4">
            {{ $users->links() }}
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
