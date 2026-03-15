@extends('layouts.main')

@section('content')
<div class="container-fluid p-4">
    <!-- Header Section -->
    <div class="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h1 class="h3 font-weight-bold text-dark mb-1" style="font-family: 'Playfair Display', serif;">Manage Transactions</h1>
            <p class="text-secondary mb-0">Pantau seluruh aktivitas transaksi di Warung Padi.</p>
        </div>
        <!-- Optional: Add filters or export buttons here if needed -->
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
                <i class="bi bi-receipt-cutoff fs-4 me-3 opacity-75"></i>
                <h5 class="fw-bold mb-0" style="letter-spacing: 0.5px;">Riwayat Transaksi</h5>
            </div>
            
            <div class="d-none d-md-block" style="width: 300px;">
                <div class="input-group">
                    <span class="input-group-text border-0 bg-white text-muted ps-3" style="border-radius: 20px 0 0 20px;">
                        <i class="bi bi-search"></i>
                    </span>
                    <input type="text" class="form-control border-0 bg-white text-dark" placeholder="Cari ID transaksi..." style="border-radius: 0 20px 20px 0; font-size: 0.9rem;">
                </div>
            </div>
        </div>

        <div class="card-body p-0">
            <div class="table-responsive">
                <table class="table table-hover align-middle mb-0">
                    <thead style="background-color: #f8f9fa;">
                        <tr>
                            <th class="ps-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">ID Transaksi</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Tanggal</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Pembeli</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Penjual</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Total</th>
                            <th class="py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Status</th>
                            <th class="text-end pe-4 py-3 text-secondary text-uppercase" style="font-size: 0.75rem; letter-spacing: 1px; font-weight: 700;">Aksi</th>
                        </tr>
                    </thead>
                    <tbody>
                        @forelse($transactions as $transaction)
                        <tr style="border-bottom: 1px solid rgba(0,0,0,0.05); transition: background-color 0.2s;">
                            <td class="ps-4 py-3 fw-bold text-dark">
                                #{{ $transaction->id_transaksi }}
                            </td>
                            <td class="text-muted">
                                <div class="d-flex align-items-center">
                                    <i class="bi bi-calendar3 me-2 text-secondary opacity-50"></i>
                                    {{ \Carbon\Carbon::parse($transaction->tanggal)->format('d M Y') }}
                                </div>
                            </td>
                            <td>
                                <div class="fw-bold text-dark" style="font-size: 0.9rem;">{{ $transaction->pembeli->nama ?? 'Unknown' }}</div>
                                <div class="small text-muted" style="font-size: 0.75rem;">Pembeli</div>
                            </td>
                            <td>
                                <div class="fw-bold text-dark" style="font-size: 0.9rem;">{{ $transaction->penjual->nama ?? 'Unknown' }}</div>
                                <div class="small text-muted" style="font-size: 0.75rem;">Penjual</div>
                            </td>
                            <td>
                                <span class="fw-bold text-success">Rp {{ number_format($transaction->harga_akhir ?? 0, 0, ',', '.') }}</span>
                            </td>
                            <td>
                                @php
                                    $statusConfig = match($transaction->status_transaksi) {
                                        'completed' => ['bg' => '#e8f5e9', 'text' => '#2e7d32', 'border' => '#c8e6c9', 'icon' => 'bi-check-circle-fill'],
                                        'pending' => ['bg' => '#fff3e0', 'text' => '#ef6c00', 'border' => '#ffe0b2', 'icon' => 'bi-hourglass-split'],
                                        'processed' => ['bg' => '#e3f2fd', 'text' => '#1565c0', 'border' => '#bbdefb', 'icon' => 'bi-arrow-repeat'],
                                        'cancelled' => ['bg' => '#ffebee', 'text' => '#c62828', 'border' => '#ffcdd2', 'icon' => 'bi-x-circle-fill'],
                                        'failed' => ['bg' => '#ffebee', 'text' => '#c62828', 'border' => '#ffcdd2', 'icon' => 'bi-exclamation-circle-fill'],
                                        default => ['bg' => '#f5f5f5', 'text' => '#616161', 'border' => '#e0e0e0', 'icon' => 'bi-circle'],
                                    };
                                @endphp
                                <span class="badge rounded-pill px-3 py-2 d-inline-flex align-items-center" 
                                      style="background-color: {{ $statusConfig['bg'] }}; color: {{ $statusConfig['text'] }}; border: 1px solid {{ $statusConfig['border'] }};">
                                    <i class="bi {{ $statusConfig['icon'] }} me-1"></i>
                                    {{ ucfirst($transaction->status_transaksi) }}
                                </span>
                            </td>
                            <td class="text-end pe-4">
                                <a href="{{ route('admin.transactions.show', $transaction->id_transaksi) }}" class="btn btn-sm btn-action bg-white text-primary border shadow-sm rounded-pill px-3" data-bs-toggle="tooltip" title="Details">
                                    Details <i class="bi bi-arrow-right ms-1"></i>
                                </a>
                            </td>
                        </tr>
                        @empty
                        <tr>
                            <td colspan="7" class="text-center py-5">
                                <div class="d-flex flex-column align-items-center justify-content-center">
                                    <img src="https://cdni.iconscout.com/illustration/premium/thumb/empty-cart-2130356-1800926.png" alt="Empty" style="width: 150px; opacity: 0.6; mix-blend-mode: multiply;">
                                    <h5 class="text-muted mt-3">Belum ada data transaksi</h5>
                                </div>
                            </td>
                        </tr>
                        @endforelse
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="card-footer bg-white border-0 py-4 px-4">
            {{ $transactions->links() }}
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

    .btn-action {
        transition: all 0.2s;
    }
    .btn-action:hover {
        background-color: #f8f9fa !important;
        transform: translateX(3px);
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
